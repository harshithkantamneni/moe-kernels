#!/usr/bin/env python
"""Model v7's decode domain MAP (registration docs/registered/2026-11-18-v7-domain-map-gh200): the 14 synthetic shapes,
the 130 map cells, v7's frozen prediction of each, the threshold-selection rule, the freeze, the plan and the rental, in
one module for the registration builder (register_v7_domain.py) and the readout (read_map.py). It imports the frozen
decode modules, v7 and the v7 blind's fit and changes none of them.

THE RULES THIS FILE OWNS:
  what       a DESCRIPTIVE measurement, not a certification: the map decides v7's decode validity domain D by the
             registered threshold-selection rule (RULE_TEXT, verbatim from Phase 0 section 6, and RULE_POINTS, how a
             program computes it); the domain blind that D feeds (Phase 2) is registered separately, later
  shapes     map_shapes.py: Phase 0's 14 geometries, synthetic names, synthetic weights; not in moe/spec.py (frozen)
  cells      Phase 0's map design (domain-phase0 p6_map.py) regenerated here exactly: families A (path twins across E),
             B (slab 2.4 to 38 MB at a fixed predicted call time), C (block-scan at E 256); one GRR main page per shape
  predict    v7 (every switch on) at the CAL2 refit, the v7 blind's fit path (v7bcommon.decode_fit), unchanged since the
             v7 gate froze it; every map cell priced before any page
  freeze     the v7 gate registration's freeze (v7 as frozen there) plus this map's own files; read_map.py refuses on
             any difference
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for _p in (str(REPO), str(REPO / "scripts"), str(REPO / "scripts" / "scoring" / "decode"),
           str(REPO / "scripts" / "scoring" / "v7"), str(REPO / "scripts" / "scoring" / "v7blind"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import map_shapes as MS  # noqa: E402
import v7bcommon as VB  # noqa: E402

V7, D2, D4, DC, FN, GV = VB.V7, VB.D2, VB.D4, VB.DC, VB.FN, VB.GV
RD = VB._X["RD"]

DATE = "2026-11-18"
NAME = f"{DATE}-v7-domain-map-gh200"
REG_JSON = f"docs/registered/{NAME}.json"
REG_TXT = f"docs/registered/{NAME}.txt"
PAGES = f"docs/registered/{DATE}-v7-domain-map-pages"
PLAN = "scripts/plans/rentalV7M-domain-map-2026-11.plan"
DRIVER = "gh200_decode_session.sh"
LAUNCHER = "scripts/vm_run_v7m.sh"
PARENT_LAUNCHER = "scripts/vm_run_v7b.sh"
VMSITE = "scripts/scoring/v7domain/vmsite"
GATE_NAME = GV.NAME
BLIND_NAME = VB.NAME
#: Lambda's list price for gpu_1x_gh200 on 2026-11-18; d2common.RATE (2.29) is the inherited registered rate
LIST_RATE = 2.59
SHUFFLE_SEED = 20261014
REPEATS = 9
#: resident CTA slots per decode tile on GH200 (132 SMs x CTAs per SM), Phase 0's table (p6_map.py)
SLOTS = {"16x64x128w4s4": 396, "64x128x64w4s3": 396, "32x64x128w4s3": 528, "32x128x64w4s3": 528}
PATHS = ("naive", "small-batch", "block-scan")
PHASE0 = "moe-kernels-work 2026-11-18 domain-phase0 (REPORT.md, p6_map.py, out/map.json, out/map.txt)"

# ================================================================== the rule

#: Phase 0 section 6, verbatim (the agent report's "Rule text, for registration before Phase 1's pages")
RULE_TEXT = (
    "For each decode path p (naive, small-batch, block-scan):\n"
    "- Take the valid map cells on p in the certified scope (every active expert in one M-tile).\n"
    "- For each cell, e = dcommon.e_dec(T_v7, T_meas).\n"
    "- e_ub = max(|T_v7/T_lo - 1|, |T_v7/T_hi - 1|), where T_lo, T_hi = T_meas ∓ 2.306 × 1.2533 × s/√n, "
    "and s is the sd of the cell's n clean repeats.\n"
    "- The candidates are the cells' predicted times. X_p is the smallest candidate X such that every candidate "
    "X' ≥ X whose set {T_v7 ≥ X'} holds at least 10 cells has:\n"
    "  - (i) every e_ub ≤ 6%;\n"
    "  - (ii) rms e ≤ 2.5%;\n"
    "  - (iii) |mean e| ≤ 1.0%;\n"
    "  - and (iv) X_p is the largest of the values this rule gives with each synthetic shape's cells removed in turn.\n"
    "- A path with no such X is outside the domain.\n"
    "- D = {decode calls whose path has an X_p, whose T_v7 ≥ X_p, and whose every active expert fits one M-tile}.\n"
    "\n"
    "Slab guard: over family B's cells under a wave, fit resid_us = a + b·W_e_MB + c·1[T=2]. If b's 95% "
    "interval excludes 0 and |b|·35 MB ≥ 1 us, D adds W_e ≥ S, with S chosen by the same rule; otherwise D "
    "has no slab condition.")

#: how read_map.py computes RULE_TEXT, every open point fixed before any page (no choice is left to a reader)
RULE_POINTS = (
    "valid: the session's power check reads 700 W (d2common.power_check; else nothing is read and D is NOT READ); the "
    "page is counted by dcommon.decode_page against this registration's page sha256 and counts sha256 (G5, inst GRR, "
    "slip-policy cell, GD4, the r7-median-run3 rebuild and G1); the cell's NATIVE row passes dcommon.row_drop (at least "
    "6 of 9 clean repeats, G2 and GD1, the G4 clock at or over 1705 MHz): an off-lock or lost cell is not valid",
    "path: the frozen prediction's path (r3_decode.geometry); scope: no active expert over BLOCK_M rows "
    "(d2common.second_tile is None; every registered map cell is in scope by construction, the test is applied anyway)",
    "T_v7: the registered prediction in us; T_meas: the rebuilt row's ms_p50 in us (the median of the clean repeats); "
    "e = dcommon.e_dec(T_v7, T_meas) (0 at or under 1 us absolute); rms, mean over a set's e",
    "s and n: the cell's clean repeats as the rebuild counts them (r7common.clean_r7 at the page's lock: status ok, a "
    "time, no lock slip, no run guard), n their number, s their sample standard deviation (n - 1); 2.306 and 1.2533 "
    "are constants (t(0.975, 8) and the median's efficiency factor), whatever n is; e_ub carries no 1 us floor (as "
    "written: e's floor is dcommon.e_dec's alone); when T_lo <= 0, or n < 2, e_ub is infinite",
    "candidates: the distinct T_v7 of the set's cells; a candidate counts only when its set {T_v7 >= X} holds at least "
    "10 cells (a candidate with fewer is neither tested nor eligible as X); X(set) is the smallest counted candidate X "
    "such that every counted candidate X' >= X passes (i), (ii) and (iii); none when no counted candidate passes so",
    "(iv): the rule above on the path's cells, and again with each synthetic shape's cells removed in turn (the 14 "
    "shapes, a shape with no cell on the path giving the full set again); X_p is the largest of these values; a path "
    "where any of them is none has no X_p and is outside the domain",
    "the slab guard: family B's valid in-scope cells (both paths) with waves under 1 (waves = the larger GEMM's live "
    "CTAs over the tile's resident slots, SLOTS); resid_us = T_v7 - T_meas in us; ordinary least squares with an "
    "intercept; a regressor constant over the fitted cells (1[T=2] has no family B cell at T 2) is left out; b's 95% "
    "interval b +/- t(0.975, n - p) x se(b), p the fitted coefficients; fitted only when n - p >= 1, else NOT FITTED "
    "and no slab condition; the guard triggers when the interval excludes 0 and |b| x 35 >= 1 us",
    "S when the guard triggers: per path with an X_p, over its cells with T_v7 >= X_p, the same rule with W_e (MB, "
    "registered per cell) in place of T_v7: candidates the distinct W_e, counted at 10 or more cells, (i) to (iii), "
    "(iv) by shape, S_p the largest; a path with an X_p and no S_p is outside the domain; D then also requires "
    "W_e >= S_p",
    "D is written per path: X_p (us) or OUTSIDE, S_p (MB) or no slab condition, and the scope condition; read_map.py "
    "prints it and writes it to domain.json and domain.txt",
)

E_UB_MAX, RMS_MAX, MEAN_MAX, MIN_CELLS = 0.06, 0.025, 0.010, 10
T975_8, MEDIAN_SE = 2.306, 1.2533
SLAB_SPAN_MB, SLAB_MIN_US = 35.0, 1.0

PHASE2 = (
    "NEXT STEP, NOT REGISTERED HERE (the plan's Phase 2, registered separately after this map is read): one blind GH200 "
    "rental on the models no decode page has timed, DeepSeek-V3, DeepSeek-V3-tp4, DeepSeek-V3-tp8 and "
    "Qwen2-57B-A14B-tp8, v7 frozen and unchanged; only their cells inside D (as this map's rule produces it, "
    "mechanically) scored, at the old bars rms 3%, worst 8%, |mean| 1.5%, own and pooled, decodeB-blind's rule (a model "
    "CERTIFIED when its own bar and the pooled bar hold); every other cell reported DESCRIPTIVE. A pass would read: v7 "
    "predicts vLLM's fused MoE decode call on GH200 within 3% rms / 8% worst on unseen models, inside D. A fail closes "
    "decode. If D is empty, no Phase 2 blind runs")


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def r6(x):
    return None if x is None else round(float(x), 6)


# ================================================================== pricing (v7 at the CAL2 refit)

def model_of(fit: dict | None = None) -> dict:
    """v7's frozen decode model: the CAL2 refit under v7 (v7bcommon.decode_fit, gate_v7.compute_decode's fit) as
    finalise_decode.frozen, D6FIX's constants, the terms and law."""
    fit = VB.decode_fit() if fit is None else fit
    c = fit["cal"]
    fz = FN.frozen(c)
    fz["model"] = {"name": "v7", "module": "scripts/r3_timing_model_v7.py", "switches": V7.V7.as_dict(),
                   "regime": "decode (v6d's law, D6FIX)", "terms": fit["terms"].as_dict(), "law": fit["law"].as_dict(),
                   "D6FIX": fit["fix"], "constants_sha256": V7.constants_sha256()}
    stats = (c.get("E1_CAL_IN_SAMPLE") or {}).get("stats")
    return {"frozen": fz, "terms": fit["terms"], "law": fit["law"], "prm": FN.params_of(fz),
            "prm_fix": VB.prm_fix_of(fit["fix"], fz.get("tile_params")),
            "cal": {"verdict": (c.get("STOP") or {}).get("verdict"), "E1_CAL_IN_SAMPLE": stats,
                    "rho": fz["rho_form"], "constants": fz["constants"]}}


def price(P: dict, model: str, T: int, counts) -> dict:
    """v7's price of one decode cell and its geometry (Phase 0's p6_map.price, unchanged)."""
    from moe.spec import MODEL_CONFIGS as MC
    cell = {"T": int(T), "counts": list(counts), "shuffle": True, "label": "x"}
    p = V7.price_decode_cell(model, cell, P["prm"], terms=P["terms"], law=P["law"], prm_fix=P["prm_fix"])
    if p is None or p.get("ms") is None:
        return {"ms": None}
    g = RD.geometry(MC[model], int(T), list(counts))
    S = SLOTS[g.tile]
    return {"ms": p["ms"], "path": g.path, "tile": g.tile, "BLOCK_M": g.bm, "A": g.A, "n": [int(x) for x in g.live_ctas],
            "slots": S, "waves": max(g.live_ctas) / S, "d6fix": "v7" in p,
            "follower": bool(p.get("v6d") and str(p["v6d"].get("naive") or "").startswith("a non-fetching"))}


def _bm(T: int) -> int:
    return RD.default_config(T, 8)["BLOCK_SIZE_M"]


def ladder(E: int, k: int, T: int, A: int):
    """A active experts (the first A), T k rows balanced; None if infeasible or an expert over BLOCK_M (Phase 0)."""
    if not (k <= A <= min(E, T * k)):
        return None
    q, r = divmod(T * k, A)
    c = [q + 1] * r + [q] * (A - r) + [0] * (E - A)
    if max(c) > T or max(c) > _bm(T):
        return None
    return c


def design(P: dict) -> list[dict]:
    """Phase 0's map (p6_map.py), regenerated: (family, model, label, T, counts, why) in Phase 0's order. Must run
    inside map_shapes.installed()."""
    from moe.spec import MODEL_CONFIGS as MC
    out = []

    def add(fam, model, label, T, counts, why):
        if counts is not None:
            out.append({"family": fam, "model": model, "label": label, "T": T, "counts": counts, "why": why})

    def a_for(model, T, target_us, k, E):
        best = None
        for A in range(k, min(E, T * k) + 1):
            c = ladder(E, k, T, A)
            if c is None:
                continue
            pr = price(P, model, T, c)["ms"] * 1e3
            if best is None or abs(pr - target_us) < abs(best[1] - target_us):
                best = (A, pr)
        return best

    for E, k in ((8, 2), (16, 4), (32, 8), (64, 8), (128, 8), (256, 8)):
        m = MS.name_of("A", E, k, 2048, 512)
        add("A", m, f"L-T1-A{k}", 1, ladder(E, k, 1, k), "naive twin base")
        for T in (2, 4, 8, 16):
            c = ladder(E, k, T, k)
            if c is not None and RD.geometry(MC[m], T, c).path != RD.NAIVE:
                add("A", m, f"MIN-T{T}", T, c, "aligned twin of L-T1 (same GEMMs)")
        for T in (32, 64, 96, 128):
            A = min(E, T * k)
            add("A", m, f"L-T{T}-A{A}", T, ladder(E, k, T, A), "BALANCED: tile / waves / long call")
    for H, I in ((1024, 384), (1536, 512), (2048, 768), (2560, 1152), (3072, 1536), (4096, 1536)):  # noqa: E741
        m = MS.name_of("B", 32, 4, H, I)
        add("B", m, "L-T1-A4", 1, ladder(32, 4, 1, 4), "naive twin base")
        add("B", m, "MIN-T4", 4, ladder(32, 4, 4, 4), "aligned twin")
        add("B", m, "MIN-T16", 16, ladder(32, 4, 16, 4), "aligned twin, T16")
        for tgt in (40, 60, 100, 200):
            b = a_for(m, 8, tgt, 4, 32)
            if b and abs(b[1] - tgt) / tgt < 0.25:
                add("B", m, f"L-T8-A{b[0]}", 8, ladder(32, 4, 8, b[0]), f"fixed call ~{tgt} us")
        add("B", m, "L-T64-A32", 64, ladder(32, 4, 64, 32), "32x64x128, over a wave")
        add("B", m, "L-T96-A32", 96, ladder(32, 4, 96, 32), "32x128x64")
        add("B", m, "L-T128-A32", 128, ladder(32, 4, 128, 32), "64x128x64")
    for H, I in ((7168, 384), (4096, 1024)):  # noqa: E741
        m = MS.name_of("C", 256, 8, H, I)
        add("C", m, "L-T1-A8", 1, ladder(256, 8, 1, 8), "naive twin base")
        add("C", m, "L-T8-A64", 8, ladder(256, 8, 8, 64), "naive, no follower")
        for T in (12, 16):
            add("C", m, f"MIN-T{T}", T, ladder(256, 8, T, 8), "block-scan twin of L-T1 (same GEMMs)")
        for T, As in ((16, (16, 32, 64, 128)), (32, (32, 128, 256)), (64, (64, 256)), (128, (128, 256))):
            for A in As:
                add("C", m, f"L-T{T}-A{A}", T, ladder(256, 8, T, A), "block-scan call-time / wave sweep")
    return out


def page_doc(model: str, cells: list[dict]) -> dict:
    return {"schema": RD.SCHEMA, "model": model, "page": f"{model}-main", "shuffle_seed": SHUFFLE_SEED,
            "decl_copies": None,
            "label": (f"SYNTHETIC (scripts/scoring/v7domain/v7mcommon.py, Phase 0's map design); the v7 domain map "
                      f"(registration {NAME}), DESCRIPTIVE; family {MS.FAMILY[model]}"),
            "cells": [{"label": c["label"], "T": c["T"], "counts": c["counts"], "arms": ["native"], "role": "LADDER"}
                      for c in cells if c["model"] == model]}


def page_path(model: str) -> str:
    return f"{PAGES}/{model}-main.json"


def page_files(cells: list[dict]) -> dict[str, str]:
    """{page path: bytes as text} for the 14 pages."""
    return {page_path(m): json.dumps(page_doc(m, cells), indent=1) + "\n" for m in MS.NAMES}


def reg_file(path: str, text: str | None = None) -> dict:
    """dcommon.decode_page's registered file record (sha256 and each cell's counts sha256)."""
    raw = (REPO / path).read_bytes() if text is None else text.encode()
    doc = json.loads(raw)
    return {"path": path, "sha256": hashlib.sha256(raw).hexdigest(), "shuffle_seed": doc["shuffle_seed"],
            "decl_copies": doc["decl_copies"],
            "cells": [{"label": c["label"], "T": c["T"], "counts_sha256": RD.counts_sha256(c["counts"]), "role": c["role"],
                       "arms": c["arms"], "shuffle": c.get("shuffle", True)} for c in doc["cells"]]}


def page_checks(texts: dict[str, str]) -> dict:
    """Each page through r3_decode.page_from (the page format and its refusals), its time estimate (9 repeats,
    r3_decode.estimated_seconds), GRR's copies and the weights they hold. Inside map_shapes.installed()."""
    from moe.spec import MODEL_CONFIGS as MC
    out = {}
    for path, text in texts.items():
        doc = json.loads(text)
        m = doc["model"]
        pg = RD.page_from(doc, MC[m], m, hashlib.sha256(text.encode()).hexdigest())
        R, calls, why = RD.rotation(MC[m], pg)
        s = RD.estimated_seconds(pg, REPEATS)
        out[m] = {"page": path, "cells": len(pg.cells), "page_from": "PASSES", "est_s": r6(s),
                  "est_min": math.ceil(s / 60), "R": R, "calls": calls, "why_R": why,
                  "weights_GB": r6(R * MC[m].num_experts * RD.expert_bytes(MC[m]) / 1e9)}
    return out


def predictions(fit: dict | None = None, P: dict | None = None) -> dict:
    """The map's frozen document: v7's model, the 130 cells (design order) with v7's price and geometry, the pages'
    texts and checks. Every cell priced before any page."""
    P = model_of(fit) if P is None else P
    with MS.installed():
        from moe.spec import MODEL_CONFIGS as MC
        cells = design(P)
        texts = page_files(cells)
        checks = page_checks(texts)
        for c in cells:
            cfg = MC[c["model"]]
            pr = price(P, c["model"], c["T"], c["counts"])
            if pr["ms"] is None:
                raise FN.Refused(f"v7 does not price {c['model']}/{c['label']}")
            st = D2.second_tile(c["model"], dict(c, role="LADDER"))
            c.update({"E": cfg.num_experts, "k": cfg.top_k, "H": cfg.hidden_size, "I": cfg.intermediate_size,
                      "W_e_MB": r6(RD.expert_bytes(cfg) / 1e6), "counts_sha256": RD.counts_sha256(c["counts"]),
                      "path": pr["path"], "tile": pr["tile"], "BLOCK_M": pr["BLOCK_M"], "A": pr["A"], "n": pr["n"],
                      "slots": pr["slots"], "waves": r6(pr["waves"]), "in_scope": st is None, "d6fix": pr["d6fix"],
                      "follower": pr["follower"], "pred_ms": pr["ms"], "pred_us": r6(pr["ms"] * 1e3)})
    for c in cells:
        del c["counts"]
    return {"frozen": P["frozen"], "CAL2 refit under v7": P["cal"], "cells": cells, "pages": texts, "checks": checks}


# ================================================================== the plan and the rental

def units(checks: dict) -> list[dict]:
    """One GRR main page per shape, nodrop, in shape order (d5common.blind_units' form, est by _rd_est's rule)."""
    return [{"label": "dm", "model": m, "page": page_path(m), "step": "r3decode", "inst": "GRR", "drop": "nodrop",
             "est_min": checks[m]["est_min"]} for m in MS.NAMES]


def unit_lines(checks: dict) -> list[str]:
    return list(D4.PRELUDE) + [
        f"{u['model']} r3decode label={u['label']} histogram={u['page']} inst={u['inst']} slip-policy=cell "
        f"g1-rule=r7-median-run3 nodrop=1 est={u['est_min']} cap={FN._cap(u['est_min'])}" for u in units(checks)]


def rental(checks: dict) -> dict:
    """d2common.rental's rule (the decode registrations'): units x 1/0.84 plus 25 min boot and setup; deadline launch +
    wall + 15 rounded to 5; hard cap 15 later; cost at the inherited d2common.RATE and at today's list price."""
    um = sum(int(x.split("est=")[1].split()[0]) for x in D4.PRELUDE) + sum(u["est_min"] for u in units(checks))
    r = D2.rental(um)
    r.update({"list_rate": f"gpu_1x_gh200 at ${LIST_RATE}/h (Lambda's list price, 2026-11-18)",
              "cost_usd_at_list": round(r["wall_min"] / 60 * LIST_RATE, 2),
              "cost_at_hard_cap_usd_at_list": round(r["hard_cap_after_launch_min"] / 60 * LIST_RATE, 2)})
    return r


def plan_text(checks: dict, fz: dict) -> str:
    c = fz["constants"]
    us = lambda v: "NOT FITTED" if v is None else f"{v * 1e3:.2f} us"  # noqa: E731
    hdr = ["# MODEL v7'S DECODE DOMAIN MAP RENTAL (DESCRIPTIVE): v7 (every switch on) frozen at the CAL2 refit, on 14",
           "# SYNTHETIC shapes (Phase 0's map: path twins across E, slab sweep at fixed call time, block-scan at E 256).",
           f"# Registration {NAME}. GENERATED by scripts/scoring/v7domain/register_v7_domain.py; do not edit by hand.",
           f"# Run by the pinned decode driver: bash {LAUNCHER} start ... --plan {PLAN} --driver {DRIVER}",
           f"# The map shapes reach moe.spec through {VMSITE} on the driver's PYTHONPATH (the launcher sets it).",
           "# Every page is nodrop. One mapping rental; the domain rule reads its pages mechanically (read_map.py).",
           f"# Frozen: T0_GR {us(c['T0_GR_ms'])}, s_naive {us(c['s_naive_GR_ms'])}, s_block {us(c['s_block_GR_ms'])}, "
           f"rho {fz['rho_form']}."]
    return "\n".join(hdr + unit_lines(checks)) + "\n"


# ================================================================== the freeze

OWN_CODE = (
    "scripts/scoring/v7domain/map_shapes.py", f"{VMSITE}/sitecustomize.py", "scripts/scoring/v7domain/v7mcommon.py",
    "scripts/scoring/v7domain/register_v7_domain.py", "scripts/scoring/v7domain/read_map.py",
    "scripts/scoring/v7blind/v7bcommon.py", "scripts/replicate_noise_floor.py",
    f"scripts/{DRIVER}", "scripts/setup_vm.sh", "scripts/vm_results_push.sh", "scripts/locked_r3.py", "scripts/r3_decode.py",
    LAUNCHER, PARENT_LAUNCHER,
    f"docs/registered/{GATE_NAME}.json", f"docs/registered/{GATE_NAME}.txt",
    f"docs/registered/{BLIND_NAME}.json", f"docs/registered/{BLIND_NAME}.txt",
)


def own_files() -> tuple:
    return tuple(page_path(m) for m in MS.NAMES) + OWN_CODE


def freeze(root: Path = REPO) -> dict:
    g = VB.gate_registration()["freeze"]
    own = {p: sha256(Path(root) / p) for p in own_files() if p not in g["files"]}
    return {"statement": ("read_map.py refuses unless every file of the v7 gate registration's freeze has its sha256 "
                          "there and v7's constants hash is the one registered there (v7 as the gate froze it, "
                          "unchanged since), and every file under 'own' has its sha256 here; the plan's sha256 is "
                          "computed by the registration builder from the same document"),
            "gate registration": {"name": GATE_NAME, "json sha256": sha256(REPO / f"docs/registered/{GATE_NAME}.json"),
                                  "files": len(g["files"]), "constants_sha256": g["constants_sha256"]},
            "own": own}


def verify_freeze(reg: dict, root: Path = REPO) -> list[str]:
    """Every difference from the registered freeze (empty: frozen)."""
    fz = reg["freeze"]
    bad = []
    gp = Path(root) / "docs" / "registered" / f"{GATE_NAME}.json"
    if not gp.is_file() or sha256(gp) != fz["gate registration"]["json sha256"]:
        bad.append(f"{GATE_NAME}.json is not the registered bytes")
    else:
        bad += GV.verify_freeze(json.loads(gp.read_text()), root=root)
    for p, want in fz["own"].items():
        f = Path(root) / p
        if not f.is_file():
            bad.append(f"{p}: missing")
        elif sha256(f) != want:
            bad.append(f"{p}: sha256 {sha256(f)[:12]} is not the registered {want[:12]}")
    if V7.constants_sha256() != fz["gate registration"]["constants_sha256"]:
        bad.append("v7's constants hash moved")
    return bad
