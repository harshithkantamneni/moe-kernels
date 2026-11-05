#!/usr/bin/env python
"""The decode stage's fifth addendum (docs/registered/2026-11-04-decode-addendum5-gh200): model v5d, its gate and the
new blind test, in one module for the registration builder, the gate, the finalisation and the blind scorer. It
imports the frozen decode modules (dcommon, d2common, d3common, d4common, finalise_decode) and v5d
(r3_timing_model_v5d, over v4d') and changes none of them: every rule here acts by swapping v5d in where the frozen
code prices a cell, by naming the new blind models and their pages where the frozen scorer reads its model list and
page paths, or by computing v5d's constants from DS's readings.

THE RULES THIS FILE OWNS:
  v5d        `v5d(terms)`: d3common.price_cell_v4dp (the one function every addendum-3 hook prices a page cell with)
             priced by r3_timing_model_v5d.price at `terms`; with NONE it is v4d' exactly
  terms      `terms_from(readings)`: v5d's constants from DS's readings (addendum 4), mechanically: SUBWAVE's r_cta and
             DRAIN's r1 per tile, W1RATE's scale per larger tile, the CALL rule's coefficients by OLS on DS's NATIVE
             call rows, ALIGN's points and SLOPE's slope. A reading that is not READ is BLOCKING (`blocking`), never a
             fallback
  gate       the new blind's launch gate (addendum 4's recorded intent, owner 2026-11-04): both published trees
             reproduce their committed scores; CAL2 under v5d reads GO by the addendum-3 stop rule; the SEEN decode
             blind re-scored under v5d reads a pooled rms under 4.0% over the committed 220 cells (a WEAK SANITY FLOOR
             on seen pages, never a test). gate_d5.py prints one GO / NO-GO line; the finalisation refuses unless GO
  blind      the new blind models (Mixtral-8x22B, Qwen2-57B-A14B, Granite-3.0-1B-A400M), their pages
             (scripts/decode_synth.py's frozen generators, unchanged), units, the owner's bars (3%, 8%, 1.5%) and plan
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d2common as D2  # noqa: E402
import d3common as D3  # noqa: E402
import d4common as D4  # noqa: E402
import dcommon as DC  # noqa: E402

RD, V4, V4D, TM = DC.RD, DC.V4, DC.V4D, DC.V4.TM
REPO = DC.REPO
V4DP = D3.V4DP
import r3_timing_model_v5d as V5D  # noqa: E402

r6 = DC.r6
DATE = "2026-11-04"
NAME = f"{DATE}-decode-addendum5-gh200"
#: the new blind's registered page lists (not timed pages): named blind5, outside addendum 4's "*decode*pages" scan,
#: whose "no decode page" means no TIMED decode page of the model (d4common.no_decode_page)
PAGES5 = f"docs/registered/{DATE}-blind5-pages"
BLIND5_PLAN = "scripts/plans/rentalD5-blind-2026-11.plan"
BLIND_DRIVER = "gh200_decode_session.sh"
DS_READINGS = "ds.readings.json"
#: DS's published session (the readings' tree) and the reading tool, named for the record; the gate reads the readings
DS_TOOL = "scripts/scoring/decode/read_ds.py"

#: the new blind models (addendum 4's names) and the two with a GR1 control page (GD2), as decodeB-blind
NEW_BLIND = D4.NEW_BLIND
NEW_CTL = ("qwen2-57b-a14b", "granite-3.0-1b-a400m")
#: the owner's bars for every NEW blind test (2026-11-04, D4-1): the CAL bars, not tightened
BARS_OWNER = {"rms_max": 0.03, "max_abs": 0.08, "mean_max": 0.015,
              "source": "owner 2026-11-04 (addendum 4, D4-1): the CAL bars (3% rms per model, 8% worst cell, 1.5% |mean|), "
                        "registered before any page of the new models"}
#: the gate's second clause (owner, 2026-11-04, recorded in addendum 4): the SEEN decode blind re-scored under v5d reads
#: a pooled rms under this over the committed score's pooled cells
RESCORE_RMS_MAX = 0.040
RESCORE_CELLS = 220
SEEN_DISCLOSURE = D4.SEEN_DISCLOSURE
WEAK_FLOOR = ("a WEAK SANITY FLOOR on SEEN pages (" + SEEN_DISCLOSURE + "; v5d's form was chosen on DS's and CAL2's pages "
              "after that diagnosis); never a test and never a certification")

# --------------------------------------------------------------------------
# v5d where the frozen scorers price a cell
# --------------------------------------------------------------------------

#: v4d''s cell price as addendum 3 registered it (captured before any v5d hook can replace it)
_PRICE_V4DP = D3.price_cell_v4dp


def price_cell_v5d(model: str, cell: dict, prm, arm: str = "native", decl_copies: int | None = None,
                   fixes: V4DP.Fixes = V4DP.ALL, terms: V5D.Terms = V5D.NONE) -> dict | None:
    """dcommon.price_cell's contract (None: NOT PRICED) with v5d at `terms` over v4d' at `fixes`."""
    from moe.spec import MODEL_CONFIGS
    cfg = MODEL_CONFIGS[model]
    declared = None if arm == "native" else cfg.num_experts * int(decl_copies or 1)
    ids = None
    if fixes.naive_order and RD.geometry(cfg, int(cell["T"]), cell["counts"], declared).path == RD.NAIVE:
        ids = V4DP.ids_of(model, cell)
    try:
        p = V5D.price(model, int(cell["T"]), cell["counts"], prm, ids=ids, fixes=fixes, terms=terms, declared=declared)
    except V4.TM.Refused:
        return None
    if prm.T0_GR_ms is not None and p["ms"] is None:
        return None
    return p


@contextlib.contextmanager
def v5d(terms: V5D.Terms):
    """Inside this block d3common's cell price (what d3common.v4dp hands every frozen scorer) is v5d's at `terms`.
    Use it around d3common.v4dp (or a d3 wrapper, which enters v4dp itself)."""
    orig = D3.price_cell_v4dp

    def price_cell_v4dp(model, cell, prm, arm="native", decl_copies=None, fixes=V4DP.ALL):
        return price_cell_v5d(model, cell, prm, arm, decl_copies, fixes, terms)
    D3.price_cell_v4dp = price_cell_v4dp
    try:
        yield
    finally:
        D3.price_cell_v4dp = orig


# --------------------------------------------------------------------------
# v5d's constants from DS's readings (mechanical; no choice after the readings)
# --------------------------------------------------------------------------

LARGER_TILES = tuple(t for t in DC.DECODE_TILES if t != V4DP.W1_TILE)


def _read(x) -> bool:
    return str((x or {}).get("verdict", "")).startswith("READ")


def call_rows(readings: dict) -> list[dict]:
    """DS's NATIVE call rows (CALLS' rows as read_ds.py recorded them), silu_and_mul and moe_sum."""
    return [r for r in (readings.get("CALLS") or {}).get("rows") or []
            if r.get("arm") == RD.NATIVE and r.get("class") in V5D.CALL_CLASSES]


def fit_call_rule(rows: list[dict]) -> dict:
    """The CALL rule's coefficients per class: ordinary least squares of the class time (us) on V5D.CALL_FORMS'
    regressors (from each row's model config and T), over every DS NATIVE row of that class. READ with rows from at
    least 3 models holding at least 3 distinct values of every non-constant regressor's source (d4common's CALLS rule);
    else NOT READ (BLOCKING). Beside: the leave-one-model-out rms (DESCRIPTIVE, the form's out-of-model check)."""
    from moe.spec import MODEL_CONFIGS
    out = {}
    for cls in V5D.CALL_CLASSES:
        rs = [r for r in rows if r["class"] == cls]
        form = V5D.CALL_FORMS[cls]
        models = sorted({r["model"] for r in rs})
        if len(models) < D4.CALL_MIN_MODELS:
            out[cls] = {"verdict": f"NOT READ: rows from {len(models)} models (needs {D4.CALL_MIN_MODELS})", "models": models}
            continue
        X = np.array([[V5D.call_regressors(MODEL_CONFIGS[r["model"]], int(r["T"]), cls)[q] for q in form] for r in rs])
        y = np.array([float(r["us"]) for r in rs])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        if not np.isfinite(beta).all() or np.linalg.matrix_rank(X) < X.shape[1]:
            out[cls] = {"verdict": "NOT READ: the fit is not finite or the design is rank-deficient", "models": models}
            continue
        res = X @ beta - y
        lomo = []
        for m in models:
            tr = np.array([r["model"] != m for r in rs])
            b, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
            lomo += list(X[~tr] @ b - y[~tr])
        out[cls] = {"verdict": "READ: OLS on " + ", ".join(form), "form": list(form),
                    "coefficients": {q: float(beta[i]) for i, q in enumerate(form)}, "rows": len(rs), "models": models,
                    "rms_us": r6(float(np.sqrt(np.mean(res ** 2)))),
                    "lomo_rms_us (DESCRIPTIVE)": r6(float(np.sqrt(np.mean(np.square(lomo)))))}
    return out


def blocking(readings: dict) -> list[str]:
    """Every reading v5d needs that DS did not READ (each one BLOCKING: v5d never falls back): POWER and the readings'
    verdict, SUBWAVE and DRAIN on every decode tile, W1RATE on every larger tile, the CALL rule for both classes,
    ALIGN's small-batch points, BLOCK and SLOPE."""
    why = []
    if (readings.get("POWER") or {}).get("verdict") != "READ":
        why.append(f"POWER: {(readings.get('POWER') or {}).get('verdict')}")
    if readings.get("verdict") != "READ":
        why.append(f"readings: {readings.get('verdict')}")
    for k, tiles in (("SUBWAVE", DC.DECODE_TILES), ("DRAIN", DC.DECODE_TILES), ("W1RATE", LARGER_TILES)):
        for t in tiles:
            x = ((readings.get(k) or {}).get("tiles") or {}).get(t)
            if not _read(x):
                why.append(f"{k} {t}: {(x or {}).get('verdict', 'absent')}")
    for c, x in fit_call_rule(call_rows(readings)).items():
        if not _read(x):
            why.append(f"CALL {c}: {x['verdict']}")
    a = readings.get("ALIGN") or {}
    if not (a.get("points") or {}).get(RD.SMALL_BATCH):
        why.append("ALIGN small-batch: no point")
    for k in ("BLOCK", "SLOPE"):
        if not _read(a.get(k)):
            why.append(f"{k}: {(a.get(k) or {}).get('verdict', 'absent')}")
    return why


def terms_from(readings: dict) -> V5D.Terms:
    """v5d's terms from DS's readings, mechanically (call `blocking` first; a missing reading raises KeyError)."""
    reg = {}
    for t in DC.DECODE_TILES:
        sw, dr = readings["SUBWAVE"]["tiles"][t], readings["DRAIN"]["tiles"][t]
        reg[t] = (None if sw.get("r_cta_GBps") is None else float(sw["r_cta_GBps"]),
                  None if dr.get("r1_GBps") is None else float(dr["r1_GBps"]))
    w1 = {t: D4.V4D_GBPS / float(readings["W1RATE"]["tiles"][t]["counter_GBps"]) for t in LARGER_TILES}
    call = {c: dict(x["coefficients"]) for c, x in fit_call_rule(call_rows(readings)).items()}
    a = readings["ALIGN"]
    align = {p: tuple((int(E), float(u)) for E, u in (a["points"].get(p) or [])) for p in V5D.ALIGN_PATHS}
    return V5D.Terms(regime=reg, w1_scale=w1, call=call, align={p: v for p, v in align.items() if v},
                     slope=float(a["SLOPE"]["slope_us"]))


def load_readings(ds_dir: Path) -> dict:
    p = Path(ds_dir) / DS_READINGS
    if not p.exists():
        raise FileNotFoundError(f"no {p}")
    return json.loads(p.read_text())


def registered_terms() -> V5D.Terms:
    """v5d's terms as this addendum registered them (docs/registered/<NAME>.json)."""
    reg = json.loads((REPO / "docs" / "registered" / f"{NAME}.json").read_text())
    return V5D.Terms.from_dict(reg["model v5d"]["terms"])


REL = 1e-9


def same_terms(a: V5D.Terms, b: V5D.Terms) -> list[str]:
    """Where two v5d term sets differ (each number to 1e-9 relative; empty: the same terms)."""
    da, db = a.as_dict(), b.as_dict()
    why = []

    def walk(x, y, path):
        if isinstance(x, dict) and isinstance(y, dict):
            if set(x) != set(y):
                why.append(f"{path}: keys {sorted(x)} against {sorted(y)}")
                return
            for k in x:
                walk(x[k], y[k], f"{path}.{k}")
        elif isinstance(x, list) and isinstance(y, list):
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
    walk(da, db, "terms")
    return why


# --------------------------------------------------------------------------
# the new blind models and their pages where the frozen scorers look
# --------------------------------------------------------------------------

def page5_path(model: str, kind: str) -> str:
    return f"{PAGES5}/{model}-{kind}.json"


@contextlib.contextmanager
def blind5():
    """dcommon.BLIND_MODELS the new blind models and dcommon.page_path their pages (every other page as registered)."""
    models, path = DC.BLIND_MODELS, DC.page_path

    def page_path(model, kind):
        if model in NEW_BLIND:
            return page5_path(model, kind)
        return path(model, kind)
    DC.BLIND_MODELS, DC.page_path = NEW_BLIND, page_path
    try:
        yield
    finally:
        DC.BLIND_MODELS, DC.page_path = models, path


def page5(model: str, kind: str) -> dict:
    """The new blind's page: scripts/decode_synth.py's frozen generator (main, ctl), unchanged, relabelled."""
    import decode_synth as SY
    d = SY.page(model, kind)
    d["label"] = (f"{d['label']}; the new blind (registration {NAME}), generated by scripts/decode_synth.py's frozen "
                  "rules for a model no decode page has timed")
    return d


PAGE_KEYS = tuple((m, "main") for m in NEW_BLIND) + tuple((m, "ctl") for m in NEW_CTL)


def page5_files() -> dict[str, str]:
    return {f"{m}-{k}.json": json.dumps(page5(m, k), indent=1) + "\n" for m, k in PAGE_KEYS}


def reg_file5(model: str, kind: str) -> dict:
    p = REPO / page5_path(model, kind)
    doc = json.loads(p.read_text())
    return {"path": page5_path(model, kind), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "shuffle_seed": doc["shuffle_seed"], "decl_copies": doc["decl_copies"],
            "cells": [{"label": c["label"], "T": c["T"], "counts_sha256": RD.counts_sha256(c["counts"]), "role": c["role"],
                       "arms": c["arms"], "shuffle": c.get("shuffle", True)} for c in doc["cells"]]}


# --------------------------------------------------------------------------
# the blind units, the blind plan and the rental
# --------------------------------------------------------------------------

def _rd_est(model: str, kind: str) -> int:
    from moe.spec import MODEL_CONFIGS
    page = RD.page_from(page5(model, kind), MODEL_CONFIGS[model], model)
    return math.ceil(RD.estimated_seconds(page, 9) / 60)


def blind_units() -> list[dict]:
    """decodeB-blind's unit form for the new models: dm (GRR, nodrop) per model, then dc (GR1, drop group gr1)."""
    out = [{"label": "dm", "model": m, "page": f"{m}-main.json", "step": "r3decode", "inst": "GRR", "drop": "nodrop",
            "est_min": _rd_est(m, "main")} for m in NEW_BLIND]
    out += [{"label": "dc", "model": m, "page": f"{m}-ctl.json", "step": "r3decode", "inst": "GR1", "drop": "gr1",
             "est_min": _rd_est(m, "ctl")} for m in NEW_CTL]
    return out


def blind_unit_tuples() -> tuple:
    """finalise_decode.plan_text's unit tuples (label, model, kind, step, inst, drop, est)."""
    return tuple((u["label"], u["model"], u["page"].rsplit("-", 1)[1][:-5], u["step"], u["inst"], u["drop"], u["est_min"])
                 for u in blind_units())


PRELUDE = D4.PRELUDE
BLIND_HDR = ["# THE DECODE STAGE'S NEW BLIND RENTAL under the fifth addendum: v5d frozen from DS's readings and CAL2, three",
             f"# models no decode page has timed (registration {NAME}). GENERATED by",
             "# scripts/scoring/decode/finalise_decode_d5.py, which refuses unless the gate reads GO; do not edit by hand.",
             "# Run by the pinned decode driver: bash scripts/vm_run.sh start ... --plan "
             f"{BLIND5_PLAN} --driver {BLIND_DRIVER}",
             "# Drop order, last first: gr1 (the two GR1 control pages); every main page is nodrop.",
             "# LAUNCH PRECONDITION (owner, 2026-11-04): scripts/scoring/decode/gate_d5.py prints GO: both published trees",
             "# reproduce their committed scores, CAL2 under v5d reads GO by the addendum-3 stop rule, AND the SEEN decode blind",
             f"# re-scored under v5d reads a pooled rms under {RESCORE_RMS_MAX:.1%} over its {RESCORE_CELLS} cells (a WEAK SANITY",
             "# FLOOR on seen pages, never a test). On NO-GO this plan is never written: decode closes as calibrated, not",
             "# certified."]


def blind_plan(fz: dict | None = None) -> str:
    import finalise_decode as FN
    hdr = list(BLIND_HDR)
    if fz:
        c = fz["constants"]
        us = lambda v: "NOT FITTED" if v is None else f"{v * 1e3:.2f} us"  # noqa: E731
        hdr.append(f"# Finalised: T0_GR {us(c['T0_GR_ms'])}, s_naive {us(c['s_naive_GR_ms'])}, s_block {us(c['s_block_GR_ms'])}, "
                   f"rho {fz['rho_form']}, bars rms {fz['bars']['rms_max']:.1%}, worst {fz['bars']['max_abs']:.1%}, "
                   f"mean {fz['bars']['mean_max']:.2%} (owner).")
    with blind5():
        return FN.plan_text(blind_unit_tuples(), hdr, census=False)


def blind_plan_units() -> list[str]:
    """The registered unit lines of the blind plan (the finalisation writes them under its header)."""
    return [ln for ln in blind_plan().splitlines() if ln and not ln.startswith("#")]


def blind_rental() -> dict:
    units = sum(int(x.split("est=")[1].split()[0]) for x in PRELUDE) + sum(u["est_min"] for u in blind_units())
    return D2.rental(units)


# --------------------------------------------------------------------------
# the new models: footprint, standing, and GD5 offline
# --------------------------------------------------------------------------

def footprint(model: str) -> dict:
    """GRR's copies (r3_decode.rotation on the main page) and the weights they hold (COMPUTED)."""
    from moe.spec import MODEL_CONFIGS
    cfg = MODEL_CONFIGS[model]
    page = RD.page_from(page5(model, "main"), cfg, model)
    R, calls, why = RD.rotation(cfg, page)
    we = RD.expert_bytes(cfg)
    return {"E": cfg.num_experts, "k": cfg.top_k, "H": cfg.hidden_size, "I": cfg.intermediate_size,
            "W_e_MB": r6(we / 1e6), "R": R, "calls": calls, "weights_GB": r6(R * cfg.num_experts * we / 1e9), "why": why}


def gd5_file(model: str, device: str = "NVIDIA_GH200_480GB") -> str:
    """The tuned-config file vLLM's get_moe_configs looks for on bf16 (dtype None, no block shape): E, N = I, the
    device name (current_platform.get_device_name() with spaces as underscores), as r3_decode.tuned_and_override asks."""
    from moe.spec import MODEL_CONFIGS
    cfg = MODEL_CONFIGS[model]
    return f"E={cfg.num_experts},N={cfg.intermediate_size},device_name={device}.json"


def gd5_offline(configs_dirs: dict[str, Path] | None = None) -> dict:
    """GD5 checked offline: whether a vLLM source tree's fused_moe/configs holds the GH200 bf16 tuned file a new model's
    lookup would load (the runtime GD5 readback on the VM still refuses any cell that received another config).
    `configs_dirs` {label: dir}; with none, the registered reading (typed in, provenance below)."""
    if configs_dirs is None:
        return GD5_REGISTERED
    out = {}
    for label, d in configs_dirs.items():
        names = {p.name for p in Path(d).glob("*.json")}
        out[label] = {m: ("TUNED FILE PRESENT: " + gd5_file(m)) if gd5_file(m) in names else "NO TUNED FILE: the default "
                      "config (tile_resolve.default_config)" for m in NEW_BLIND}
    return out


#: GD5 offline (2026-11-04, build-D5): two vLLM source trees on the laptop (the pinned 0.27.1 wheel is not installed off
#: the VM): ~/code/vllm-contrib-scratch/vllm54959/vllm-main at c1d8fad (main, 2026-11-03; 339 tuned files) and
#: ~/Desktop/research/spec-decode-kernels/vllm at fafca38 (2026-03-28). Neither holds any file for device_name
#: NVIDIA_GH200_480GB; the files for these (E, N) are H200, H100 fp8 and AMD MI300X / MI325X only. get_moe_configs reads
#: one exact file name (no device fallback); the published CAL2 and blind pages bear that out: Mixtral-8x7B (E 8, N 14336)
#: has an NVIDIA_H200 file and GD5 passed on every GH200 decode page of it
GD5_REGISTERED = {
    "label": "COMPUTED offline (build-D5, 2026-11-04) on two local vLLM source trees; the VM's runtime GD5 readback binds",
    "files looked up": {m: gd5_file(m) for m in NEW_BLIND},
    "vllm main c1d8fad (2026-11-03)": {m: "NO TUNED FILE: the default config" for m in NEW_BLIND},
    "vllm fafca38 (2026-03-28)": {m: "NO TUNED FILE: the default config" for m in NEW_BLIND},
    "nearest files present (not GH200)": {
        "mixtral-8x22b": ["E=8,N=16384,device_name=AMD_Instinct_MI300X.json", "E=8,N=16384,device_name=AMD_Instinct_MI325X.json"],
        "qwen2-57b-a14b": ["E=64,N=2560,device_name=NVIDIA_H200.json", "E=64,N=2560,device_name=NVIDIA_H200,dtype=fp8_w8a8.json",
                           "E=64,N=2560,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8.json"],
        "granite-3.0-1b-a400m": []},
    "lookup": "one exact file name per (E, N, device, dtype); no device fallback (the GH200 CAL2 and blind pages of "
              "Mixtral-8x7B, which has an NVIDIA_H200 file, passed GD5)",
    "verdict": "PASS offline: no new model has a GH200 tuned file, so vLLM runs default_config(T, E), the tile v5d prices",
    "not the pinned version": ("neither tree checked is the pinned vLLM 0.27.1 (its wheel is not installed off the VM), so "
                               "this is not GD5 itself: the VM's runtime GD5 readback would still have bound, refusing any "
                               "cell that received another config"),
}


# --------------------------------------------------------------------------
# checks the registration computes
# --------------------------------------------------------------------------

#: DS's readings as committed beside the other decode scores (read_ds.py's output on the published DS session)
DS_READINGS_COMMITTED = f"scripts/scoring/decode/{DS_READINGS}"


def committed_readings() -> dict:
    return json.loads((REPO / DS_READINGS_COMMITTED).read_text())


def reduction(terms: V5D.Terms) -> dict:
    """COMPUTED: v5d with every term off is v4d' on every registered CAL and blind main-page NATIVE cell (geometry
    prices, no constants), and with `terms` on, every naive cell's GEMM prices are v4d''s (the GEMM terms act on the
    aligned paths only)."""
    from moe.spec import MODEL_CONFIGS
    prm = D4.ds_prm()
    n = same = naive = naive_same = 0
    for m in DC.CAL_MODELS + DC.BLIND_MODELS:
        for c in DC.page_doc(m, "main")["cells"]:
            if "native" not in c.get("arms", ["native"]):
                continue
            ids = None
            if RD.geometry(MODEL_CONFIGS[m], int(c["T"]), c["counts"]).path == RD.NAIVE:
                ids = V4DP.ids_of(m, c)
            try:
                a = V4DP.price(m, int(c["T"]), c["counts"], prm, ids=ids)
            except TM.Refused:
                continue
            b = V5D.price(m, int(c["T"]), c["counts"], prm, ids=ids, terms=V5D.NONE)
            n += 1
            same += int(a["body_ms"] == b["body_ms"] and a["bytes"] == b["bytes"])
            if a["path"] == RD.NAIVE:
                t = V5D.price(m, int(c["T"]), c["counts"], prm, ids=ids, terms=terms)
                naive += 1
                naive_same += int(all(t["gemm"][g]["gemm_ms"] == a["gemm"][g]["gemm_ms"] for g in TM.GEMMS))
    return {"cells": n, "identical with every term off": same, "naive cells": naive,
            "naive cells keeping the v4d' GEMM prices with the terms on": naive_same,
            "verdict": "REDUCES" if n and same == n and naive_same == naive else "DIFFERS"}


def new_model_reach(terms: V5D.Terms) -> dict:
    """COMPUTED from the new models' configs and their registered page lists (no page of theirs is read): per model the
    paths, tiles and wave classes of its main cells, the align points v5d uses at its E (measured, interpolated or held),
    the small-batch iterations, and each call class's regressors against DS's range."""
    from moe.spec import MODEL_CONFIGS
    rows = call_rows(committed_readings())
    rng = {}
    for cls in V5D.CALL_CLASSES:
        bs = [V5D.call_regressors(MODEL_CONFIGS[r["model"]], int(r["T"]), cls)["B_MB"] for r in rows if r["class"] == cls]
        ks = sorted({MODEL_CONFIGS[r["model"]].top_k for r in rows if r["class"] == cls})
        rng[cls] = {"B_MB": [r6(min(bs)), r6(max(bs))], "k": ks}
    out = {"DS call range": rng}
    for m in NEW_BLIND:
        cfg = MODEL_CONFIGS[m]
        E = cfg.num_experts
        paths, its, waves, Ts = {}, set(), {}, set()
        for c in page5(m, "main")["cells"]:
            g = RD.geometry(cfg, int(c["T"]), c["counts"])
            paths[g.path] = paths.get(g.path, 0) + 1
            Ts.add(int(c["T"]))
            if g.path == RD.SMALL_BATCH:
                its.add(V5D.iterations(g.numel, E))
            if g.path != RD.NAIVE:
                S = 132 * (4 if g.tile in ("32x64x128w4s3", "32x128x64w4s3") else 3)
                for i, gm in enumerate(RD.GEMMS):
                    n = g.live_ctas[i]
                    k = f"{g.tile} {gm} " + ("sub-wave" if n <= S else "crossing" if n < 2 * S else "multi-wave")
                    waves[k] = waves.get(k, 0) + 1
        al = {}
        for p in V5D.ALIGN_PATHS:
            if not paths.get(p):
                continue
            pts = dict(terms.align.get(p) or ())
            es = sorted(pts)
            how = ("MEASURED point" if E in pts else "HELD at the nearest measured E" if (not es or E < es[0] or E > es[-1])
                   else "INTERPOLATED between measured E")
            al[p] = {"E": E, "us": r6(V5D.align_us(terms, p, E)), "rule": how, "measured_E": es}
        calls = {}
        for cls in V5D.CALL_CLASSES:
            b = [V5D.call_regressors(cfg, T, cls)["B_MB"] for T in sorted(Ts)]
            calls[cls] = {"B_MB": [r6(min(b)), r6(max(b))], "k": cfg.top_k, "k in DS": cfg.top_k in rng[cls]["k"],
                          "largest B over DS's largest": r6(max(b) / rng[cls]["B_MB"][1])}
        out[m] = {"E": E, "k": cfg.top_k, "H": cfg.hidden_size, "I": cfg.intermediate_size, "main cells by path": paths,
                  "GEMMs by tile and wave class (aligned paths)": dict(sorted(waves.items())),
                  "small-batch iterations": sorted(its), "align": al, "calls": calls}
    return out
