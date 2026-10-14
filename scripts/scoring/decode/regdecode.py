"""The decode stage's two registrations, built from committed files (register_decode.py writes them):
2026-10-14-decodeC-cal-gh200 (the CAL rental: OLMoE and Mixtral) and 2026-10-14-decodeB-blind-gh200
(the blind rental: Phi, Qwen1.5, JetMoE, deepseek-v2-lite, Granite-3B), the second FROZEN now and
finalised mechanically from the first's score (finalise_decode.py, which refuses on a STOP)."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

import dcommon as DC

REPO = DC.REPO
RD, V4D = DC.RD, DC.V4D
r6 = DC.r6
PRICE_PER_H = 2.29
#: rentals 4 to 7's wall over units (boot, setup, pushes)
WALL_FACTOR = 1 / 0.84
SETUP_WALL_MIN = 15
POWER_DRAWS = 4000
POWER_SEED = 20261014
#: the power simulation's assumed per-cell noise (relative sd of a decode cell's 6-of-9 median):
#: rental 3's sigma_page, doubled for the sub-50 us cells (ASSUMED, registered)
POWER_SIGMA = 0.0018
POWER_SIGMA_SMALL = 0.0036

DESIGN = ("design-decode DESIGN.md (2026-10-14, design only, read-only against 80514a6) and the owner's decisions of "
          "2026-10-14: DD1 NATIVE alone certifies, SHARED's declaration arm (DECL) an optional lever on CAL models; DD2 "
          "vLLM's naive path (T 1 to 3) in scope; DD3 a summary-statistics-only extraction at B 1 to 128, or the "
          "design's fallback (rental 5's fitted shapes extrapolated) when the raw traces are unavailable, registered; "
          "DD4 the multi-call CUDA graph (GRR) is the primary timer, a one-call graph (GR1) the control; DD5 provisional "
          "bars rms <= 3%, worst <= 8%, |mean| <= 1.5%; DD6 decode runs after the tile stage (built now, rented after "
          "tile rental B)")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def page_record(model: str, kind: str) -> dict:
    rel = DC.page_path(model, kind)
    doc = DC.page_doc(model, kind)
    return {"path": rel, "sha256": sha256(REPO / rel), "shuffle_seed": doc["shuffle_seed"],
            "decl_copies": doc["decl_copies"],
            "cells": [{"label": c["label"], "T": c["T"], "counts_sha256": RD.counts_sha256(c["counts"]),
                       "role": c["role"], "arms": c["arms"], "shuffle": c.get("shuffle", True)} for c in doc["cells"]]}


def units_of(units) -> list[dict]:
    out = []
    for label, model, kind, step, inst, drop, est in units:
        out.append({"label": label, "model": model, "page": f"{model}-{kind}.json", "step": step, "inst": inst,
                    "drop": drop, "est_min": est})
    return out


def minutes(units) -> dict:
    est = sum(u[6] for u in units) + 3 + 2 + (6 if units is DC.CAL_UNITS else 0)
    wall = round(est * WALL_FACTOR + SETUP_WALL_MIN)
    return {"units_min": est, "wall_min": wall, "cost_usd": round(wall / 60 * PRICE_PER_H, 2),
            "rate": f"gpu_1x_gh200 at ${PRICE_PER_H}/h",
            "rule": "units plus prelude 3 and calibrate 2 (and the CAL census 6), x 1/0.84 (rentals 4 to 7) plus 15 min boot and setup"}


def nominal_cells(models, kinds=("main",)) -> list[dict]:
    """DESIGN-NOMINAL v4d at every page cell: for scale, levers and power only."""
    out = []
    for m in models:
        for kind in kinds:
            doc = DC.page_doc(m, kind)
            for c in doc["cells"]:
                row = {"model": m, "page": f"{m}-{kind}", "label": c["label"], "T": c["T"], "role": c["role"]}
                for form in V4D.RHO_FORMS:
                    p = DC.price_cell(m, c, DC.nominal_params(form))
                    row[f"ms_nominal_{form}"] = None if p is None else r6(p["ms"])
                    if form == "fixed" and p is not None:
                        row.update(path=p["path"], tile=p["tile"], A=p["A"], L=p["L"], P=p["P"],
                                   dead_ctas=p["dead_ctas"], sub_wave=p["sub_wave"],
                                   weight_MiB=round(p["weight_bytes"] / 2 ** 20, 2))
                out.append(row)
    return out


def power(cells: list[dict], bars: dict, scenarios: dict) -> dict:
    """P(the E1-DEC bar HOLDS) per scenario, Monte Carlo on the CPU in float64: measured = truth x (1 +
    N(0, sigma_cell)), truth = the nominal prediction x (1 + bias(cell)); bias per scenario."""
    rng = np.random.default_rng(POWER_SEED)
    pred = np.array([c["ms_nominal_fixed"] for c in cells if c.get("ms_nominal_fixed")])
    small = np.array([c["ms_nominal_fixed"] < 0.05 for c in cells if c.get("ms_nominal_fixed")])
    sig = np.where(small, POWER_SIGMA_SMALL, POWER_SIGMA)
    out = {}
    for name, fn in scenarios.items():
        bias = np.array([fn(c) for c in cells if c.get("ms_nominal_fixed")])
        truth = pred * (1 + bias)
        holds = 0
        for _ in range(POWER_DRAWS):
            meas = truth * (1 + rng.standard_normal(pred.size) * sig)
            d = pred - meas
            e = np.where(np.abs(d) * 1e3 <= DC.FLOOR_US, 0.0, d / meas)
            rms, worst, mean = math.sqrt(float(np.mean(e * e))), float(np.max(np.abs(e))), float(np.mean(e))
            holds += rms <= bars["rms_max"] and worst <= bars["max_abs"] and abs(mean) <= bars["mean_max"]
        out[name] = r6(holds / POWER_DRAWS)
    return {"draws": POWER_DRAWS, "seed": POWER_SEED, "cells": int(pred.size), "sigma_cell": POWER_SIGMA,
            "sigma_cell_under_50us": POWER_SIGMA_SMALL, "p_holds": out, "computed": "CPU float64 (numpy)",
            "label": "DESIGN-NOMINAL predictions; sigma ASSUMED"}


SCENARIOS = {
    "the model is the truth": lambda c: 0.0,
    "T0_GR off by +5 us (the riskiest assumption)": lambda c: -0.005 / c["ms_nominal_fixed"] if c.get("ms_nominal_fixed") else 0.0,
    "T0_GR off by +1 us": lambda c: -0.001 / c["ms_nominal_fixed"] if c.get("ms_nominal_fixed") else 0.0,
    "+1% everywhere": lambda c: -0.01 / 1.01,
    "+2% everywhere": lambda c: -0.02 / 1.02,
    "+5% on sub-wave cells": lambda c: (-0.05 / 1.05) if c.get("sub_wave") else 0.0,
}

#: per-model provenance of the blind models (build-decode review 5)
SEEN_MODELS = {
    "phi-3.5-moe": "SEEN-MODEL: R3 pages published (rental 7, R3's tile); the decode call, tile and instrument BLIND",
    "qwen1.5-moe-a2.7b": "SEEN-MODEL: R3 pages published (R3's tile); the decode call, tile and instrument BLIND",
    "jetmoe-8b": "SEEN-MODEL: R3 pages published (R3's tile); the decode call, tile and instrument BLIND",
    "granite-3.0-3b-a800m": "SEEN-MODEL: R3 pages published (R3's tile, the launch floor); the decode call, tile "
                            "and instrument BLIND",
    "deepseek-v2-lite": "SEEN-SHAPE (the tile stage's D7: on a published H200 page, its GEMMs Qwen1.5's); the decode "
                        "call, tile and instrument BLIND",
}


def coldness(models) -> dict:
    """GRR's reuse distance per model: R copies x A_min x W_e MiB between two reads of one copy,
    against the GH200's 60 MiB L2 (two 30 MiB partitions)."""
    from moe.spec import MODEL_CONFIGS
    out = {}
    for m in models:
        cfg = MODEL_CONFIGS[m]
        page = RD.load_page(REPO / DC.page_path(m, "main"), cfg, m)
        R, calls, _w = RD.rotation(cfg, page)
        a_min = min(sum(1 for v in c.counts if v) for c in page.cells)
        we = RD.expert_bytes(cfg) / 2 ** 20
        out[m] = {"R": R, "calls_per_graph": calls, "A_min": a_min, "W_e_MiB": round(we, 1),
                  "reuse_distance_MiB": round(R * a_min * we, 1), "l2_MiB": 60,
                  "how": ("COLD BY VOLUME ONLY: R = 1, every call reads copy 0; between two reads of a slab the "
                          f"call streams at least A_min x W_e = {a_min * we:.0f} MiB, {a_min * we / 60:.0f}x the L2"
                          if R == 1 else
                          f"COLD BY ROTATION: call j reads copy j mod {R} (COPIES checks {R} distinct addresses); "
                          f"{R * a_min * we:.0f} MiB between two reads of one copy")}
    return out


GATES = {
    "V-BIN": "bincount(ids) == counts, k distinct experts per token (feasible, realize_counts); a failure refuses the cell",
    "V-ALIGN": "the spy's readback of one eager call (sorted ids None iff naive; num_tokens_post_padded; sentinel count P; "
               "expert_ids[:L] ascending; sorted length the capped buffer; the launch grid fused_moe_grid's; USE_TD False) "
               "equals r3_decode.expected_readback; a mismatch refuses the cell; any refused CAL cell is a STOP",
    "GD5": "the config the kernel received is tile_resolve.default_config(T, E), vLLM finds no tuned file and no override",
    "GD4": "every captured call's output equals the eager output bit for bit; a failure VOIDS the page. It covers "
           "the CHECK-PHASE capture only: each timed repeat recaptures the same calls on the same tensors and is "
           "not compared again",
    "COPIES": "(GRR) calls 0..R-1 are each spied in the check phase and must read R distinct w1 copies "
              "(r3_decode.check_copies); a failure refuses the cell. At R = 1 the call is cold by volume only "
              "(see coldness)",
    "GD1": "GPU per replay >= 3 x the host enqueue per replay (launch_floor's host probe, before and after); a failing "
           "cell is excluded (G2)",
    "GD2": "GR1 - GRR per call on matched cells; the CAL band [p05 - 1 us, p95 + 1 us] is frozen into the blind "
           "registration; a STOP when CAL's median is outside [-1, 10] us, the band is wider than 8 us, or the band "
           "leaves [-2, 11] us; a blind GR1 page with over 20% of its cells outside the band VOIDS that model's GRR page",
    "GD3": "E240 - GR1 in [0, 10] us on cells GPU-bound eager (C + F240 > 1.25 H): DESCRIPTIVE",
    "G1": "rental 7's r7-median-run3 rebuilt from cells.csv (6 of 9 clean; the 10% non-clean cap); a FAIL puts the "
          "page out (NOT SCORED)",
    "G2": "GD1, and the duty timer's host flag",
    "G4": "the cell's clock >= 1705 MHz",
    "G5": "the page's decode-page sha256 and every row's counts sha256 are the registered ones",
    "V-SRC": "the VM's installed fused_moe.py and moe_align_block_size.py are v0.27.1's bytes "
             "(moe/instrumented/UPSTREAM_DECODE.json); else the page is refused before any cell",
}

AMBIGUITIES = {
    "1 'each relative error floored at 1 us'": "read as: an absolute error at or under 1 us counts as 0; otherwise "
                                               "e = (pred - meas) / meas (v4d.rel_err)",
    "2 T0_GR + s_path": "s_small_GR is folded into T0_GR: every cell has exactly one path, so the three CAL "
                        "constants are T0_GR (the small-batch constant), s_naive_GR and s_block_GR (offsets); "
                        "block-scan occurs on OLMoE at T 128 in CAL, so s_block_GR is fitted, not assumed",
    "3 the wgmma decode tile (T 97..128, 64x128x64 w4)": "v4d prices it with the tile stage's finalised v4 constants "
                                                         "(c_128, F_wg, accounting, F form, census occupancy; "
                                                         "scripts/scoring/tileA/finalise.selections). With none, those "
                                                         "cells are NOT PRICED and out of E1-DEC and the fit",
    "4 the hot expert's second tile": "priced at a full slab re-read (sigma 1). The L2-survival rival and the FILL-r "
                                      "rival are STRUCK before any page (build-decode review 5): neither is priced; "
                                      "G3-D fill is FILL0's test and MINCOVER past BM shows the second tile's cost "
                                      "as a residual",
    "5 naive repeats": "vLLM launches the naive path token-major (expert_ids = topk_ids.view(-1)); v4d lists the "
                       "rows expert-ascending, which leaves the read total and the throughput sum unchanged and "
                       "moves only the window means and the tail set. A repeated expert's slab is fetched once "
                       "(the later CTA in launch order hits L2); NAIVE-NODEDUP is the rival",
    "6 G3-D clusters": "(model, T) for shuffle and label, (model, tile band) for fill; d in microseconds net of "
                       "v4d's own predicted difference",
    "7 GD2's void": "a blind GR1 page outside the band voids the model's GRR page (the instruments disagree on "
                    "that model, so the primary cannot be trusted there)",
    "8 GRR rotation": "R = ceil(240 MiB / (A_min W_e)) copies per page, A_min the page's least coverage; calls per "
                      "graph the least multiple of R at or over 8",
    "9 CERTIFIED": "a blind model is CERTIFIED when its own E1-DEC bar holds and the pooled bar holds",
    "10 the rho margin": "ABSOLUTE: scaled is selected only when its sub-wave rms is lower than fixed's by at least "
                         "0.001, i.e. 0.1 percentage points of rms (not 0.1% of fixed's rms)",
    "11 an unfitted offset": "never 0: a path with no valid CAL row has no offset, its cells are NOT PRICED, and "
                             "any NOT PRICED CAL cell is a STOP",
    "12 NOT PRICED is never silent": "the CAL scorer REFUSES without --tile-scores (the tile stage's selections; "
                                     "every CAL main page holds T 97..128 cells), STOPs on any NOT PRICED CAL "
                                     "cell, and the blind scorer lists NOT PRICED cells per model",
    "13 the pooled blind bar": "computed over the models that survive (not voided by GD2 or G3-D, with at least "
                               "one valid priced cell); the score names them (pooled_over)",
    "14 blind G3-D INCONCLUSIVE": "a TOST that is NOT EQUIVALENT (the interval leaves the margin, whether by a "
                                  "shift or by width) voids the model's cells; NOT SCORED (under 2 clusters) "
                                  "voids nothing",
    "15 the 1 us floor": "kept as a STEP and stated: |pred - meas| <= 1 us counts 0, above it the full relative "
                         "error counts (at a 12 us cell an 8% error, 0.96 us, counts 0; 9%, 1.08 us, counts 9%)",
    "16 sub-wave": "a cell is sub-wave when some GEMM's live CTAs are under SMs x its occupancy, the occupancy the "
                   "census's (tile rental A's census score) where it exists, else the shared-memory rule: the rho "
                   "selection's cell set moves with the census",
}


def cal() -> dict:
    files = {f"{m}-{k}.json": page_record(m, k) for _l, m, k, *_r in DC.CAL_UNITS}
    nom = nominal_cells(DC.CAL_MODELS)
    cov = json.loads((REPO / DC.PAGES / "coverage_stats.json").read_text())
    up = json.loads(RD.UPSTREAM.read_text())
    return {
        "registered": f"{DC.DATE}, before any decode-stage page",
        "name": DC.NAMES["cal"],
        "status": "REGISTERED BEFORE ANY PAGE",
        "design": DESIGN,
        "tool": "scorer scripts/scoring/decode/score_cal.py (dcommon: fit_and_select, g3d, gd2_band, gd3, stop_rule); "
                "pages scripts/decode_synth.py; model scripts/r3_timing_model_v4d.py",
        "what": "the decode stage's CAL rental on OLMoE-1B-7B and Mixtral-8x7B: v4d's three CAL constants and its rho "
                "selection, the instruments' agreement (GD2, GD3), G3-D, the DECL lever, the counter pages, the "
                "replicate (sigma_page and the blind bars), and the STOP rule for the blind rental",
        "units": units_of(DC.CAL_UNITS),
        "drop_order": "last first: rep, decl, ncu, e240; every main and ctl unit is nodrop",
        "census": {"tiles": list(DC.DECODE_TILES), "models": list(DC.CAL_MODELS + DC.BLIND_MODELS),
                   "rule": "the tile stage's tile_census.py, compile only; its occupancy replaces the shared-memory "
                           "rule in v4d (tcommon.use_census); a path mismatch is a STOP"},
        "pages": {"dir": DC.PAGES, "files": files,
                  "label": "SYNTHETIC (decode_synth.py); DW cells EXTRAPOLATED"},
        "instruments": {
            "GRR": "one CUDA graph of `calls` calls, call j on copy j mod R, no flush; per call = replay / calls "
                   "(launch_floor.graph_call, launch_floor.mirror_time_cell at duty 0.25, lock 1710)",
            "GR1": "a one-call graph, the 240 MiB flush outside it",
            "E240": "the eager call, the 240 MiB flush",
            "timing": {"duty": RD.DUTY, "warmup_ms": RD.WARMUP_MS, "cell_budget_ms": RD.CELL_BUDGET_MS,
                       "trials": RD.TRIALS, "repeats": 9, "lock_mhz": DC.LOCK_MHZ}},
        "model": {
            "name": "v4d (scripts/r3_timing_model_v4d.py) over v4 and the pinned v3 machinery",
            "formula": "T = T0_GR + s_path + sum_g [gemm_g + dead_g] + B_other / bw",
            "card_constants": {k: v for k, v in V4D.CARD.items()},
            "card_constants_label": "SEEN (rental 6's CAL 8x7B v2 fit)",
            "cal_constants": ["T0_GR_ms", "s_naive_GR_ms", "s_block_GR_ms"],
            "fit": "v4d.fit_constants: weighted least squares on the relative error over the valid NATIVE cells of "
                   "the two CAL main pages (the replicate excluded), cells the wgmma tile cannot price excluded",
            "rho_forms": {"fixed": f"{V4D.RHO_GBPS} GB/s per resident CTA (v3's CORES rho)",
                          "scaled": "x the tile's weight bytes in flight per CTA over R3's (16x64x128 s4: 2.0)"},
            "rho_selection": f"the lower CAL rms on sub-wave cells; fixed unless scaled's rms is lower by "
                             f"{DC.RHO_MARGIN * 100:.1f} percentage points of rms (ABSOLUTE, {DC.RHO_MARGIN})",
            "fill": "FILL0: padded and masked rows cost 0 (G3-D fill tests it)",
            "tile_stage": "v4's finalised Params (tileA/finalise.selections), census occupancy per decode tile; "
                          "REQUIRED: score_cal.py refuses without --tile-scores",
            "struck_rivals": "L2-survival and FILL-r, struck before any page (build-decode review 5)",
        },
        "gates": GATES,
        "coldness": coldness(DC.CAL_MODELS),
        "tests": {
            "E1-CAL-IN-SAMPLE": {"label": "IN-SAMPLE (the CAL constants are fitted on these cells)",
                                 "rule": "e_dec on every valid NATIVE main-page cell at the selected rho, against the "
                                         "provisional bars; a FAIL is a STOP", "bars": DC.BARS_PROVISIONAL},
            "G3-D shuffle": "UNIFORM counts with rows shuffled (U-T) against unshuffled (US-T), T 8, 32, 128",
            "G3-D label": "the same count multiset on another active set: MIN-T (experts 0..k-1) against L-T-A<k>, T 4, 16, 64",
            "G3-D fill (FILL0, a model test)": "BALANCED cells of one A at two T in one tile band",
            "G3-D margin": f"max({DC.G3D_US} us, {DC.G3D_REL:.2%} x the median reference time); TOST at 90%, "
                           "clusters (model, T)",
            "rep": "OLMoE's main page again: sigma_page = rms(rep / main - 1) / sqrt 2 over valid NATIVE pairs; the "
                   "blind bars are tightened from it (dcommon.tighten), never loosened",
            "DECL (DESCRIPTIVE)": "DECL - NATIVE per BALANCED cell at T >= BM against v4d's dead-CTA difference",
            "BYTES-D (DESCRIPTIVE)": "the counter pages' dram__bytes_read per GEMM against v4d's read bytes",
        },
        "stop_rule": "STOP (the blind rental does not run) when any: a nodrop CAL unit not counted (missing, NOT "
                     "SCORED, G1 FAIL or GD4 VOID); the fit NOT FITTED under both rho forms; the tile stage's "
                     "selections missing or any valid CAL NATIVE cell NOT PRICED; E1-CAL-IN-SAMPLE fails the "
                     "provisional bars; G3-D shuffle or label not EQUIVALENT; any CAL cell refused by V-ALIGN, GD5 "
                     "or COPIES; GD2 FAIL (median outside [-1, 10] us, band over 8 us wide or leaving [-2, 11] us). "
                     "Else GO. Refused and failed cells are taken out of G1's non-clean share before it is counted "
                     "(dcommon.drop_dead_cells), so a refusal cannot fail the page",
        "trace_statistics": {"primary": cov["primary_extract"],
                             "fallback": "rental 5's fitted DW shapes (synth_fit.json, fitted at B >= 256) "
                                         "EXTRAPOLATED to B 1 to 128 (coverage_stats.json, DW-T cells)",
                             "coverage_stats_sha256": sha256(REPO / DC.PAGES / "coverage_stats.json"),
                             "finding": "the DW shapes cut coverage by at most about 9 points against uniform "
                                        "(OLMoE B 16 0.80 against 0.89); the ladder, not the trace draws, is the lever"},
        "upstream": {"record": "moe/instrumented/UPSTREAM_DECODE.json", "sha256": sha256(RD.UPSTREAM),
                     "files": {k: v["sha256"] for k, v in up["files"].items()},
                     "design_check": up["design_check"]},
        "nominal": "DESIGN-NOMINAL: v4d at the tile stage's nominal wgmma constants (c_128 900, F_wg 520 / 979) and "
                   f"T0_GR {DC.NOMINAL['T0_GR_ms'] * 1e3:.0f} us, s 0; for scale, levers and power only",
        "nominal_cells": nom,
        "power": power(nom, DC.BARS_PROVISIONAL, SCENARIOS),
        "rental": minutes(DC.CAL_UNITS),
        "labels": {"time": "IN-SAMPLE (CAL)", "pages": "SYNTHETIC", "trace draws": "EXTRAPOLATED",
                   "card constants": "SEEN", "everything published": "SEEN"},
        "scorer_ambiguities": AMBIGUITIES,
        "seen_data": ["every published page (rentals 1 to 7) is SEEN", "rental 6's CAL fit (c, bw, eager T0)",
                      "rental 2's host enqueue 0.33 to 0.36 ms per R3 call (GD1's motivation)"],
    }


def blind() -> dict:
    files = {f"{m}-{k}.json": page_record(m, k) for _l, m, k, *_r in DC.BLIND_UNITS}
    nom = nominal_cells(DC.BLIND_MODELS)
    return {
        "registered": f"{DC.DATE}, before any decode-stage page",
        "name": DC.NAMES["blind"],
        "status": "FROZEN; FINALISED MECHANICALLY from the CAL score (scripts/scoring/decode/finalise_decode.py), which "
                  "refuses unless the CAL stop rule reads GO",
        "design": DESIGN,
        "tool": "scorer scripts/scoring/decode/score_blind.py; finalisation scripts/scoring/decode/finalise_decode.py",
        "what": "E1-DEC BLIND: v4d with its CAL constants frozen, on five models no decode page has timed",
        "units": units_of(DC.BLIND_UNITS),
        "drop_order": "last first: the two GR1 control pages (gr1); every main page is nodrop",
        "pages": {"dir": DC.PAGES, "files": files, "label": "SYNTHETIC (decode_synth.py); DW cells EXTRAPOLATED"},
        "finalisation": {
            "refuses": "unless cal.score.json's STOP reads GO",
            "frozen_from_cal": ["T0_GR_ms", "s_naive_GR_ms", "s_block_GR_ms", "the selected rho form",
                                "the GD2 band", "sigma_page and the tightened bars", "the tile stage's v4 Params"],
            "writes": "the blind predictions per cell (v4d and every rival) and scripts/plans/rentalD-blind-2026-10.plan",
        },
        "statistic": "e_dec = (T_v4d - T_meas) / T_meas per valid NATIVE cell, 0 at or under 1 us absolute",
        "rule": "E1-DEC: pooled over the five models AND per model at the bars finalised from the CAL replicate "
                "(the provisional rms <= 3%, worst <= 8%, |mean| <= 1.5% when there is no replicate); a model is "
                "CERTIFIED when its own bar and the pooled bar hold",
        "bars_provisional": DC.BARS_PROVISIONAL,
        "rivals": {"eager": "v4d at v3's eager constants (T0 43 us, s_small, s_block)",
                   "roofline": "T0_GR + A W_e / bw",
                   "line": "a + b (T k), fitted on the CAL NATIVE cells (pooled)",
                   "OLS": "ridge on standardised [A W_e / bw, waves w1 + w2, ln(1 + dead CTAs), naive, block-scan] "
                          "over the CAL NATIVE cells (the tile stage's ols_fit rule)",
                   "NAIVE-NODEDUP": "v4d with every naive row's slab fetched",
                   "rho other": "v4d at the unselected rho form",
                   "exclusion": "rental 7's rule (tcommon.rival_rule): lever cells |T_R / T_v4d - 1| >= 2%, at "
                                "least 6; EXCLUDED when R's rms there is over 2% and over v4d's + 3 se"},
        "gates": GATES,
        "coldness": coldness(DC.BLIND_MODELS),
        "g3d_blind": "shuffle and label on every blind main page at the CAL's margin rule; a FAIL voids that model's cells",
        "riskiest_assumption": "T0_GR + s_path is one card constant that transfers across models; on the small "
                               "models it is 20 to 60% of the call, and the CAL's two models cannot separate it "
                               "from H, I or the kernel count",
        "nominal": "DESIGN-NOMINAL as in decodeC-cal",
        "nominal_cells": nom,
        "power": power(nom, DC.BARS_PROVISIONAL, SCENARIOS),
        "rental": minutes(DC.BLIND_UNITS),
        "labels": {"time": "BLIND (the decode call, its tile and its instrument: no decode page exists)",
                   "pages": "SYNTHETIC", "everything published": "SEEN",
                   "per_model": SEEN_MODELS},
        "scorer_ambiguities": AMBIGUITIES,
        "seen_data": ["every published page (rentals 1 to 7) is SEEN", "the CAL rental's pages, once scored"],
    }


BUILDERS = {"cal": cal, "blind": blind}


def text_cal(d: dict) -> list[str]:
    r = d["rental"]
    return [f"CAL rental: {r['units_min']} min of units, about {r['wall_min']} min wall, ${r['cost_usd']}",
            f"power at the provisional bars (DESIGN-NOMINAL, sigma ASSUMED): {d['power']['p_holds']}"]


def text_blind(d: dict) -> list[str]:
    r = d["rental"]
    return [f"blind rental: {r['units_min']} min of units, about {r['wall_min']} min wall, ${r['cost_usd']}",
            f"power at the provisional bars (DESIGN-NOMINAL, sigma ASSUMED): {d['power']['p_holds']}"]


TEXT = {"cal": text_cal, "blind": text_blind}
