#!/usr/bin/env python
"""Writes rental 4's six registered JSON files (docs/registered/2026-10-06-rental4-*).

    python scripts/scoring/rental4/register.py [--out-dir docs/registered] [--check]

Every number is computed here from committed files (the timing model, the published pages,
the instrumented copy's and the tools' own constants) or is a typed-in input with its
provenance and its label (CAL, SEEN, SEEN-fitted, SEEN third-party). `--check` recomputes and
compares with the committed JSON and text (exit 1 on any difference) instead of writing; the
`code_pins` block is excluded from the comparison, as in rental 3.

Design: scratchpad design-r4/DESIGN.md, corrected by design-r4-review/REVIEW.md and then by
reanalysis/REANALYSIS.md section 4; the later document wins. The owner's decisions of
2026-10-06 (1 the NATIVE-only page gates, 2 the perturbation tolerance, 3 gpu-benches on the VM,
4 A3 cut, 5 the instrumented copy) are recorded where they bind. The independent review of the
build (scratchpad build-r4-review/REVIEW.md) is applied before any page: the eviction trio and
its registration are CUT (Triton 3.7.1 drops the hint on pipelined cp.async loads), and its
other fixes are marked `review_fix` where they land.

LEAKAGE, said once for all six: every published page is SEEN, and every constant typed in
below was fitted or chosen after seeing published pages (labelled per entry). No rental-4 page
exists when this runs. A file's `seen_data` names what of it is SEEN.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r4common as C4  # noqa: E402

REPO = C4.REPO
PUB = REPO / "results" / "published"
REG = REPO / "docs" / "registered"
r6 = C4.r6
R2S = PUB / "2026-10-02-nvidia_gh200_480gb-rental2-session" / "results"
R1S = PUB / "2026-10-01-nvidia_gh200_480gb-rental1-session" / "results"

# --------------------------------------------------------------------------
# typed-in inputs, each with its provenance
# --------------------------------------------------------------------------

#: the CAL models' in-kernel SHARED - NATIVE dead-CTA rows (model, GEMM, N_dead SHARED, N_dead
#: NATIVE, hidden window u in us, in-kernel gap in us): scratchpad offset/work/deadfit.json,
#: derived offline from the published counter pages (rescore/decomp/inkernel scripts there;
#: the derivation is not in this repository). SEEN.
DEADFIT_CAL = [
    ("mixtral-8x7b", "w1", 31360, 3584, 66.34650626864, 9.11),
    ("mixtral-8x7b", "w2", 4480, 512, 183.788491403296, 0.01),
    ("mixtral-8x22b", "w1", 35840, 4096, 98.75412930864, 4.27),
    ("mixtral-8x22b", "w2", 6720, 768, 209.714589835296, 0.34),
    ("qwen2-57b-a14b", "w1", 44640, 4960, 58.24460050864, 23.99),
    ("qwen2-57b-a14b", "w2", 31248, 3472, 34.713425419296, 21.51),
    ("olmoe-1b-7b", "w1", 17856, 1984, 33.93888322864, 6.31),
    ("olmoe-1b-7b", "w2", 17856, 1984, 15.268851595296, 13.66),
]
#: D2: per-GEMM d fitted on all 16 GEMMs (CAL and SEEN held-out), principles/work/dead_ratio.txt
D2 = {"d_w1": 0.93, "d_w2": 1.03, "kappa": 0.30, "label": "SEEN-fitted on all 16 GEMMs (8 CAL, 8 held-out), principles/work/dead_ratio.py"}
M_RIVAL = {"d": 1.333, "kappa": 0.5, "label": "the registered model M: DEAD_CTA_NS 1.333, k_w 0.5 (CAL, r3_timing_model)"}
#: the k-step laws, parameters fitted on the SEEN tp2 G = 64 knob pages (reanalysis/work/kstep_law.txt)
LAWS = {
    "MVA2": {"eta": 0.832, "z0": 1503.751, "z1": -1251.132,
             "form": "closed-queue MVA: occ CTAs, pipe service W(BK)/eta, delay Z = z0 + z1 BK/64 (>= 0)",
             "label": "SEEN-fitted (reanalysis kstep_law, 10 points, rms 1.28%): three parameters on four distinct (BK, occ), description not mechanism"},
    "LK": {"eta": 0.805, "L_k": 241.912, "form": "max(W(BK)/eta, L_k): a constant floor per k-step",
           "label": "SEEN-fitted (REVIEW.md section 2; reanalysis rms 5.53%)"},
    "PS": {"c64": 344.1, "form": "c = c64 BK/64: the per-CTA main loop invariant in BK (the design's P_s)",
           "label": "CAL c (8x7B counters); the design's zero-refit principle"},
    "LITTLE": {"eta": 0.805, "L_k": 241.912,
               "form": "max(W(BK)/eta, L_mem/(stages - 1)), L_mem = 3 L_k: Little per stage",
               "label": "REVIEW.md section 2 rival; L_k SEEN-fitted"},
    "OCC": {"c0": 302.0, "lam": 200.0, "form": "c0 + lambda/occ at BK 64 only (the review's occupancy rule)",
            "label": "SEEN-fitted on three occupancies (PRINCIPLES.md section 4); says nothing off BK 64"},
}
#: RRZE-HPC/gpu-benches' published GH200 outputs, parsed by scripts/gpubench.py's registered
#: rules (gpu-latency/gh200.txt, gpu-stream/gh200.txt at master 23e586dd): SEEN third-party
#: numbers, cited and not redistributed (GPL-3.0 files)
RRZE_GH200 = {"near_l2_ns": 141.616, "far_l2_ns": 253.384, "dram_ns": 346.414, "triad_gbps": 3783.0,
              "read_gbps": 2775.0, "init_gbps": 3944.0, "clock_mhz": 1980.0,
              "label": "SEEN third-party (RRZE-HPC/gpu-benches master 23e586dd, GH200, 1980 MHz), parsed by gpubench.py's rules"}
BW_CAL = 3598.0   # 8x7B fit's bw (CAL), GB/s at 1710
STAGE_BYTES = (32 + 64) * 64 * 2   # one CTA k-step's A + B tiles at 32x64x64 bf16


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# dead: the copies contrast A1, A2 and the NATIVE null control
# --------------------------------------------------------------------------

def _model_gap(d, ka, row):
    _m, _g, NS, NN, u, _gap = row
    return max(0.0, NS * d * 1e-3 - ka * u) - max(0.0, NN * d * 1e-3 - ka * u)


def _grid(fn, xs, ys):
    best = None
    for x in xs:
        for y in ys:
            c = fn(x, y)
            if best is None or c < best[0]:
                best = (c, x, y)
    return best


def fit_D() -> dict:
    """D refit on the CAL rows (the offset study's grid), with se by finite differences."""
    import numpy as np
    xs = [0.3 + 0.005 * i for i in range(441)]
    ys = [0.005 * i for i in range(401)]
    cost, d, ka = _grid(lambda a, b: sum((_model_gap(a, b, r) - r[5]) ** 2 for r in DEADFIT_CAL), xs, ys)
    res = np.array([_model_gap(d, ka, r) - r[5] for r in DEADFIT_CAL])
    h = 1e-3
    J = np.array([[(_model_gap(d + h, ka, r) - _model_gap(d - h, ka, r)) / (2 * h),
                   (_model_gap(d, ka + h, r) - _model_gap(d, ka - h, r)) / (2 * h)] for r in DEADFIT_CAL])
    s2 = float((res ** 2).sum()) / (len(DEADFIT_CAL) - 2)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(J.T @ J)))
    return {"d_ns": r6(d), "kappa": r6(ka), "se": [r6(se[0]), r6(se[1])],
            "cal_rms_us": r6(math.sqrt(float((res ** 2).mean())))}


def fit_slot() -> dict:
    """SLOT: d_g = t / occ_g, kappa, refit on the same CAL rows (design-r4 r4_deltas.py)."""
    occ = {"w1": 5, "w2": 4}

    def cost(t, ka):
        return sum((max(0, r[2] * t / occ[r[1]] * 1e-3 - ka * r[4])
                    - max(0, r[3] * t / occ[r[1]] * 1e-3 - ka * r[4]) - r[5]) ** 2 for r in DEADFIT_CAL)
    c, t, ka = _grid(cost, [1.0 + 0.01 * i for i in range(900)], [0.005 * i for i in range(300)])
    return {"t_ns_slot": r6(t), "kappa": r6(ka), "cal_rms_us": r6(math.sqrt(c / len(DEADFIT_CAL)))}


CONTRASTS = {"A1": ("qwen2-57b-a14b-tp8", 15), "A2": ("olmoe-1b-7b", 15)}


def dead() -> dict:
    D = fit_D()
    S = fit_slot()
    occ = {"w1": 5, "w2": 4}
    rivals = {
        "D": lambda g: (D["d_ns"], D["kappa"]),
        "D2": lambda g: (D2["d_" + g], D2["kappa"]),
        "M": lambda g: (M_RIVAL["d"], M_RIVAL["kappa"]),
        "K5": lambda g: (D["d_ns"], 0.5),
        "K0": lambda g: (D["d_ns"], 0.0),
        "SLOT": lambda g: (S["t_ns_slot"] / occ[g], S["kappa"]),
    }
    pred = {}
    for name, (model, copies) in CONTRASTS.items():
        E = C4.experts(model)
        row = {}
        for k, f in rivals.items():
            per = {g: C4.exposure_us(model, E * copies, g, *f(g), occ[g])
                   - C4.exposure_us(model, E * 9, g, *f(g), occ[g]) for g in ("w1", "w2")}
            row[k] = {**{g: r6(v) for g, v in per.items()}, "total": r6(sum(per.values()))}
        row["FIXED"] = {"w1": 0.0, "w2": 0.0, "total": 0.0}

        def tot(d, ka):
            return sum(C4.exposure_us(model, E * copies, g, d, ka, occ[g])
                       - C4.exposure_us(model, E * 9, g, d, ka, occ[g]) for g in ("w1", "w2"))
        sd = (tot(D["d_ns"] + D["se"][0], D["kappa"]) - tot(D["d_ns"] - D["se"][0], D["kappa"])) / 2
        sk = (tot(D["d_ns"], D["kappa"] + D["se"][1]) - tot(D["d_ns"], D["kappa"] - D["se"][1])) / 2
        pred[name] = {"model": model, "copies": [9, copies], "declared_slots": [E * 9, E * copies],
                      "dead_ctas": {"9": C4.dead_counts(model, E * 9), str(copies): C4.dead_counts(model, E * copies)},
                      "rivals": row, "sigma_D_us": r6(math.hypot(sd, sk))}
    return {
        "registered": f"{C4.DATE}, before any rental-4 page",
        "name": C4.NAMES["dead"],
        "status": "REGISTERED BEFORE ANY PAGE",
        "design": "design-r4 DESIGN.md stage 1 part A, with REVIEW.md section 3 (D2 beside D, the NATIVE null control, both noise values, the key on timed units only) and REANALYSIS.md section 4; A3 (mixtral-8x7b-tp2, the kappa lever) CUT by the owner on 2026-10-06 (decision 4)",
        "tool": "scorer scripts/scoring/rental4/score_dead.py",
        "pages": {"path": "gaps-<card>-<model>-<label>/private_weight_reference/*/report.json, labels c9 and c15, the G = 8 page at treads 1..9",
                  "contrasts": {k: {"model": m, "copies": [9, c]} for k, (m, c) in CONTRASTS.items()},
                  "declaration_check": "each page's recorded copies_declared must equal its label's (9 or 15); else the contrast is NOT SCORED",
                  "label": "every c15 page is NOT JOINABLE TO COUNTER BYTES (no counter page declares 15; validate_counter_plan): its directory carries DECLARED_COPIES.txt saying so"},
        "statistic": {"Delta_n": "[SHARED(n) - NATIVE(n)] at c15 minus the same at c9, us, ms_p50 off each page's treads_table, same board, G = 8",
                      "treads": [3, 4, 5, 6, 7, 8, 9],
                      "Delta": "median over n = 3..9 of Delta_n (the dead count is n-free, so Delta_n is flat in n)",
                      "alignment": "A_n = [align(SHARED, n) - align(NATIVE, n)] at c15 minus at c9, median over each page's align_probe repeats, us; the registered statistic is Delta - median_n A_n (the alignment kernel's growth with the declared slots, the design's confound); raw Delta printed"},
        "fit_D_CAL": D, "fit_SLOT_CAL": S, "D2": D2, "M": M_RIVAL,
        "predictions_us": pred,
        "noise": {"sigma_noise_us": 0.9, "sigma_noise_basis": "the design's: 1.25 x 2 x 0.18% x 0.5 ms / sqrt 7 (sigma_page 0.18% SEEN, rental 3 rep8)",
                  "sigma_noise_resolution_us": 0.45, "resolution_basis": "REVIEW.md section 3: SEEN SHARED - NATIVE on qwen2-tp8 spans 29.6 to 30.4 us over six pages; every verdict is also printed at 0.45 us, the registered one is at 0.9",
                  "sigma_rival": "sigma_D (the CAL fit's se, by finite differences) for D; 0 for every other rival"},
        "z": "(Delta_meas - Delta_pred) / sqrt(sigma_noise^2 + sigma_rival^2)",
        "rules": {"D": "FAILS if |z_D| > 3 on A1 or A2; else HOLDS on the contrasts measured",
                  "D2": "the same rule, scored beside D (REVIEW.md section 3)",
                  "M": "EXCLUDED if |z_M| > 3 on A1 or A2 (separation 7 to 8 us)",
                  "FIXED": "EXCLUDED if Delta > 3 sigma_noise above 0 on A1 or A2",
                  "K0": "EXCLUDED if |z| > 3 on A1 or A2 (K0 differs from D on A1 by its w1 term)",
                  "K5": "NOT TESTED: K5 equals D on A1 and A2; A3, its lever, is cut",
                  "SLOT": "EXCLUDED if |z| > 3; if D and SLOT both survive, 'D against SLOT UNDECIDED' (they separate by 0.4 and 2.0 us)",
                  "not_separated": {"pairs": ["D/SLOT", "D/D2"], "rule": "printed NOT SEPARATED whatever the pages read: the predictions differ by 0.4 to 2.2 sigma_noise on A1 and A2, so no reading can tell them apart", "review_fix": "build-r4-review section 3"},
                  "per_gemm_d": "PRINTED, not a verdict: d_w2 = Delta_A1 / dN_w2(A1) and d_w1 = (Delta_A2 - dN_w2(A2) d_w2) / dN_w1(A2), ns (both declarations past the hidden window under D, so the windows cancel)"},
        "null_control": {"what": "NATIVE(c15, n) - NATIVE(c9, n) per tread n = 3..9: NATIVE declares E slots on both pages, so its call is the same call",
                         "rule": "PASS when |median_n| <= 3 sqrt(2) x 0.18% x median_n T_NATIVE(c9, n); a FAIL turns every A verdict INCONCLUSIVE (board drift between the pages)"},
        "gates": "R3's VALIDITY gates by rental 3's two views (ALL and CLEAN, rental 2's addendum rule 3); V5 does not gate timed pages; a page whose V1 failed is unusable in both",
        "missing": "a contrast with a page missing or dropped is NOT SCORED; the drop-groups a1 / a2 keep each c9 / c15 pair together",
        "seen_data": ["D's and SLOT's fits are CAL (in-kernel counters of the four CAL models); D2 is SEEN-fitted on all 16 GEMMs",
                      "sigma 0.18% and the 0.45 us spread are SEEN (rental 3)", "no qwen2-tp8 or OLMoE page at 15 copies exists: both contrasts are BLIND"],
    }


# --------------------------------------------------------------------------
# occlaw: B', the k-step law at BLOCK_K and num_stages on OLMoE G = 64 counters
# --------------------------------------------------------------------------

def seen_kstep_points() -> list[dict]:
    """The SEEN tp2 G = 64 knob pages' per-page c (r4common.page_c, n >= 2, F held), the points
    the laws were fitted on; each registered law's residual on them is printed."""
    pages = sorted(R2S.glob(f"*-{C4.CARD}-mixtral-8x7b-tp2-at*-r3-counters/lock1710/r3c-g64.json"))
    pages += sorted(R1S.glob(f"*-{C4.CARD}-mixtral-8x7b-tp2-atile-r3-counters/lock1710/r3c-g64.json"))
    out = []
    for f in pages:
        p = json.loads(f.read_text())
        for g in ("w1", "w2"):
            r = C4.page_c(p, g, range(2, 10), C4.F_CTA[g])
            if r:
                out.append({"page": f.parts[-3].split("480gb-")[-1].replace("-r3-counters", ""), "gemm": g,
                            "BK": r["block_k"], "stages": r["num_stages"], "occ": r["occupancy"][0], "c": r6(r["c"])})
    return out


def occlaw() -> dict:
    from moe.spec import MODEL_CONFIGS
    cfg = MODEL_CONFIGS["olmoe-1b-7b"]
    K = {"w1": cfg.hidden_size, "w2": cfg.intermediate_size}
    pts = seen_kstep_points()
    fitcheck = {}
    for law, p in LAWS.items():
        e = [C4.law_c(law, p, x["BK"], x["stages"], x["occ"]) / x["c"] - 1 for x in pts
             if C4.law_c(law, p, x["BK"], x["stages"], x["occ"]) is not None]
        fitcheck[law] = {"points": len(e), "rms": r6(math.sqrt(sum(v * v for v in e) / len(e))) if e else None}
    base = C4.KNOBS["k64s4"]
    preds = {}
    for page, knob in C4.KNOBS.items():
        if page == "k64s4":
            continue
        preds[page] = {}
        for g, gi in (("w1", 0), ("w2", 1)):
            preds[page][g] = {law: r6(C4.ratio_pred(law, p, knob, C4.OCC[page][gi], base, C4.OCC["k64s4"][gi],
                                                    K[g], C4.F_CTA[g])) for law, p in LAWS.items()}
    return {
        "registered": f"{C4.DATE}, before any rental-4 page",
        "name": C4.NAMES["occlaw"],
        "status": "REGISTERED BEFORE ANY PAGE",
        "design": "REVIEW.md part B' (sections 2, 3 and 7: every page carries the registered P_s, the max form, Little and the occupancy-rule values; pass if |e| <= 2% on the floor-bound treads n >= 4, per GEMM) with REANALYSIS.md 1a (MVA2 added, k64s8 added at occupancy 2)",
        "tool": "scorer scripts/scoring/rental4/score_occlaw.py; r4common.page_c, law_c, ratio_pred",
        "pages": {"path": "<date>-<card>-olmoe-1b-7b-<label>-r3-counters/lock1710/r3c-g64.json",
                  "base": "k64s4", "knobs": {k: {"block_k": v[0], "num_stages": v[1]} for k, v in C4.KNOBS.items() if k != "k64s4"},
                  "design_check": "each page's design block (block_k, num_stages) must equal its label's; else NOT SCORED"},
        "statistic": {"c_page": "per (page, GEMM): c by least squares of sm__cycles_elapsed.avg = a_arm + ceil(N_live / 132) (S c + F_g) over NATIVE and SHARED cells n = 4..9 whose GEMM reads DRAM at <= 0.5 x 4022 GB/s (floor-bound), F_g held at CAL (w1 520, w2 979 clk); fewer than 3 such cells: NOT SCORED",
                      "r_meas": "(S_knob c_knob + F) / (S_base c_base + F), the knob page over the same-board k64s4 page",
                      "e": "r_meas / r_pred - 1 per law"},
        "laws": LAWS, "laws_on_seen_points": {"points": pts, "rms_by_law": fitcheck},
        "occupancy": {"registered": {k: {"w1": v[0], "w2": v[1]} for k, v in C4.OCC.items()},
                      "basis": "the SEEN cubins of these configs (moe.instrumented.cubin over results/published: registers and shared, the occupancy calculator); 252 of 252 GEMM records agree with ncu (PRINCIPLES.md)",
                      "rekey": "the scorer reads each page's recorded launch__occupancy_limit_* minimum; where it differs from the registered value, the page's predictions are recomputed at the recorded occupancy by the registered law functions, and the page is labelled RE-KEYED"},
        "predicted_ratio": preds,
        "rules": {"per_page": "a law is consistent on a (page, GEMM) when |e| <= 2%",
                  "law": "FALSIFIED by any scored (page, GEMM) beyond 2%; SELECTED when consistent on every scored (page, GEMM), at least 8 of the 12 scored, and every other law FALSIFIED; else UNDECIDED with the survivors printed",
                  "OCC": "scored on the BK 64 pages only (k64s6, k64s2, k64s3, k64s8)",
                  "printed": "each law's rms over the scored (page, GEMM)"},
        "gates": "the counter page's V-gates by rental 2's two views; V6, V7 and V10 do not gate (cycles are read per GEMM, and V7 / V10 are the byte claims' gates)",
        "one_test": "B' and the stamps K block share configurations (BK 32 / 64 / 128 and s8 on OLMoE G = 64): they are ONE test of the k-step law, reported together, never counted as two confirmations (build-r4-review section 5)",
        "labels": {"k64s4": "SEEN: OLMoE G = 64 at BK 64 s4 has a published page (2026-09-29), so the base page's cells are SEEN",
                   "knobs": "BLIND-CALMODEL: no OLMoE G = 64 page at these knobs; OLMoE is a CAL model",
                   "in_sample": "MVA2 is in-sample at the k32s4, k64s4 and k64s8 configurations (fitted on tp2 pages of those configs)",
                   "review_fix": "build-r4-review section 3"},
        "seen_data": ["every law's parameters are SEEN-fitted on rental 1 and 2's tp2 G = 64 pages (labelled per law)",
                      "the occupancy table is read off SEEN cubins", "the k64s4 base is SEEN (2026-09-29 OLMoE G = 64 page); the six knob pages are BLIND-CALMODEL"],
    }


# --------------------------------------------------------------------------
# perturb: the gate of every instrumented unit (owner decision 2)
# --------------------------------------------------------------------------

def perturb() -> dict:
    import instr_probe as IP

    from moe import instrumented as I
    return {
        "registered": f"{C4.DATE}, before any rental-4 page",
        "name": C4.NAMES["perturb"],
        "status": "REGISTERED BEFORE ANY PAGE: a gate, no verdict of the study's own",
        "design": "owner decision 2 (2026-10-06): median <= 1%, worst <= 2%, identical occupancy limit; REANALYSIS.md section 4 (plain vs instrumented, stamps on, hints none)",
        "tool": "scripts/instr_probe.py --mode perturb (the gate, on the VM); scorer scripts/scoring/rental4/score_perturb.py (recomputes it off perturb.json)",
        "tolerance": IP.PERTURB_TOL,
        "variants": "one per instrumented unit of scripts/plans/rental4-2026-10.plan after the perturb unit (the driver writes them: gh200_model_session.sh --print-variants --plan ...)",
        "cells": {"stamps_unit": "its own cells (G x arm x n)", "bytes_unit": f"its G at treads {list(IP.BYTE_GATE_TREADS)}, every arm"},
        "copy_spec": {"stamps_unit": "the unit's stamps, no hint", "bytes_unit": "every instrument off (the hint is the unit's treatment; the gate certifies the copy itself)"},
        "method": {"timer": "a CUDA event pair around each fused_moe_kernel launch, the same wrapper for plain (moe.instrumented.timed_plain) and copy (install)",
                   "flush": "an L2 flush (moe.bench.timing.L2Flusher at the device's size) before every timed call, OUTSIDE the event pairs, on both sides, as R3's pages flush (review_fix: build-r4-review section 2)",
                   "cache": "each perturb and stamps unit compiles into its own TRITON_CACHE_DIR (the driver sets $R/triton-cache; the probe sets the same), so no earlier unit's compile hides one (review_fix: bug G1)",
                   "host": f"torch.cuda._sleep of {IP.SLEEP_AHEAD_MS} ms ahead of each burst, so no launch waits on Python",
                   "bursts": f"{IP.REPS} bursts of {IP.CALLS} calls each way, alternating order; one untimed call first under the same kernel (no compile inside a burst)",
                   "ratio": "r = median over bursts of the burst-median plain ms / the same for the copy, per (cell, GEMM)"},
        "rule": {"median": "median over the variant's (cell, GEMM) of |r - 1| <= 1%", "worst": "max |r - 1| <= 2%",
                 "occupancy": "plain and copy PAIRED BY CONFIG (moe.instrumented.config_key: BM, BN, BK, G, stages, warps, MUL_ROUTED_WEIGHT, top_k), every plain config with a copy, and per config equal registers (the cubin's EIATTR_REGCOUNT), shared (Triton's metadata) and CTAs per SM, read off each launch's CompiledKernel (review_fix: section 2)",
                 "sass": "per config, the ALL-OFF copy's SASS (cuobjdump -sass, addresses, encodings and line info stripped: moe.instrumented.cubin.normalize_sass) equals the plain kernel's; unread FAILS (review_fix: section 1)",
                 "hint": "a unit asking an eviction hint FAILS unless its compiled PTX carries L2::cache_hint on the A/B copies (cubin.hint_in_ptx); the counter child refuses such a page too (private_weight_reference.counter_hint_refusal), with no manifest (review_fix: E1)",
                 "upstream": f"the installed vLLM fused_moe.py's sha256 is {I.UPSTREAM['file_sha256']} and its excerpt's {I.UPSTREAM['excerpt_sha256']}; else FAIL",
                 "use": "an instrumented unit runs only when its line in gate.env reads PASS (the driver refuses it in 0 minutes otherwise); a FAIL leaves its pages unwritten, never written and voided"},
        "printed": ["each variant's per-cell ratios and per-config registers and shared, and the copy's SASS bracket checks (cubin.sass_checks), which the stamps scorer gates on the unit's own SASS"],
        "eviction_hints": {"status": "CUT from rental 4 before any page (build-r4-review E1): Triton 3.7.1's LowerLoops carries eviction_policy into AsyncCopyGlobalToLocalOp and its cp.async lowering emits .ca / .cg only, so every pipelined rental-4 config compiles a hinted copy to the all-off kernel",
                           "deferred": "an L2 access-policy-window design (cudaAccessPolicyWindow on the stream, host side, the kernel unchanged) is DEFERRED and unregistered; the instrumented copy keeps its EVICT_A / EVICT_B parameters, and any hinted unit is refused by the hint leg above"},
        "seen_data": ["none: the copy has never run; the tolerance is the owner's"],
    }


# --------------------------------------------------------------------------
# stamps: F split, the k-step law per iteration, tail CTAs, dead CTAs
# --------------------------------------------------------------------------

def stamps() -> dict:
    cyc = {k: r6(RRZE_GH200[k] * C4.LOCK_MHZ / 1e3) for k in ("near_l2_ns", "far_l2_ns", "dram_ns")}
    from moe.spec import MODEL_CONFIGS
    cfg = MODEL_CONFIGS["olmoe-1b-7b"]
    K = {"w1": cfg.hidden_size, "w2": cfg.intermediate_size}
    kpred = {}
    for page, knob in (("stk64s4", (64, 4, "k64s4")), ("stk32s4", (32, 4, "k32s4")),
                       ("stk128s4", (128, 4, "k128s4")), ("stk64s8", (64, 8, "k64s8"))):
        kpred[page] = {}
        for g, gi in (("w1", 0), ("w2", 1)):
            occ = C4.OCC[knob[2]][gi]
            kpred[page][g] = {"occ": occ, "iterations": K[g] // knob[0],
                              "T_iter_cycles": {law: r6(occ * C4.law_c(law, p, knob[0], knob[1], occ))
                                                for law, p in LAWS.items() if C4.law_c(law, p, knob[0], knob[1], occ) is not None}}
    rho = 11.48e9
    tail = {"R1": r6(STAGE_BYTES / rho * 1e9), "R3L": r6(STAGE_BYTES / (3 * STAGE_BYTES / 1433e-9) * 1e9),
            "R3I": r6(STAGE_BYTES / (3 * STAGE_BYTES / 331e-9) * 1e9)}
    disp = {str(occ): {"DISP": 0.995, "MIX": r6(0.53 + 264.0 / (C4.SMS * occ)), "SLOT": r6(4.05 / occ)}
            for occ in (5, 4, 2)}
    import r3_timing_model as TM
    fs = {}
    for model, ns in (("mixtral-8x22b", (1,)), ("mixtral-8x7b", (1, 2))):
        old = TM.set_model(model)
        try:
            for n in ns:
                for g, occ in (("w1", 5), ("w2", 4)):
                    live = TM.live_rows(n) * TM.GEOMETRY[g].npn
                    wave = C4.SMS * occ
                    tail_n = live - (live // wave) * wave if live // wave >= 1 else 0
                    fs[f"{model} n{n} {g}"] = {"live": int(live), "wave": wave, "tail_ctas": int(tail_n),
                                               "FS_ns": r6(STAGE_BYTES * tail_n / (BW_CAL * 1e9) * 1e9) if tail_n else None}
        finally:
            TM.set_model(old)
    classes = {"PS": ["PS"], "LK=LITTLE": ["LK", "LITTLE"], "MVA2~OCC": ["MVA2", "OCC"]}
    return {
        "registered": f"{C4.DATE}, before any rental-4 page",
        "name": C4.NAMES["stamps"],
        "status": "REGISTERED BEFORE ANY PAGE",
        "design": "REANALYSIS.md section 4 (the stamps block: F split, Z(BK, occ), rho, d0 and L) with sections 1b, 1c, 1f; owner decision 5 (the instrumented copy, gated by perturb)",
        "tool": "scripts/instr_probe.py --mode stamps; scorer scripts/scoring/rental4/score_stamps.py; r4common.load_stamps and the readouts there",
        "instrument": "moe/instrumented/fused_moe_instr.py: clock64 (SM cycles) and globaltimer (ns) per CTA; per-iteration clock64 with stamps=iter. A unit whose perturb gate is not PASS has no page",
        "prologue_stamp": {"statement": "the prologue stamp sits before the k-loop in the source, which is before the software pipeliner's peeled fill in the compiled kernel; no source point lies after the fill and before iteration 0's first wait without changing the loop's logic, so the fill is NOT separated: it falls inside iteration 0 (the loop phase)",
                           "consequence": "the fill / epilogue split (F_g = prologue + epilogue against CAL F_w1 520, F_w2 979) is WITHDRAWN; F reads only Delta_epi = epilogue(w2) - epilogue(w1), in which the stamps' own stores cancel; K drops the first two marks",
                           "review_fix": "build-r4-review section 1"},
        "sass_precondition": {"rule": "a unit is scored only when its OWN compiled kernels' SASS (dumped by the stamps unit, cuobjdump -sass) passes moe.instrumented.cubin.sass_checks for its readout: F: a clock read before the first LDG (num_tokens) on both GEMMs and, on w2, a clock read between the loop-exit BAR and the topk_weights LDG; K and T: at least 2 clock reads inside the k-loop; D: a clock read before the first LDG; else NOT SCORED (r4common.sass_precondition)",
                              "review_fix": "build-r4-review section 1 (the bracket check is a scoring precondition, not a print)"},
        "flush": "an L2 flush before every measured call, outside the stamped launches, as R3 flushes",
        "pages": "<date>-<card>-instr-<model>-<label>/stamps.json and stamps/*.npy",
        "F": {"unit": "stf (mixtral-8x7b, G = 8, n 2, 4, 6, NATIVE, stamps=cta)",
              "readout": "per launch, the median over live CTAs of epilogue = end_clk - epi_clk; Delta_epi = median over n of [epilogue(w2) - epilogue(w1)], cycles",
              "rivals_cycles_at_1710": {"NEAR": cyc["near_l2_ns"], "FAR": cyc["far_l2_ns"], "DRAM": cyc["dram_ns"]},
              "rivals_source": "this rental's gpubench constants (cycles at the lock) when its unit ran and parsed; else the published RRZE GH200 plateaus (SEEN third-party, below)",
              "reading": "the w2 epilogue's one exposed topk_weights load (REANALYSIS.md 1b: SASS shows it is not hoisted)",
              "rule": "r4common.nearest (linear): the nearest rival when nearer than half the gap to the next, else BETWEEN",
              "printed": "per GEMM the median prologue (before the fill), loop (fill included) and epilogue cycles; no F_g split is read (prologue_stamp above)"},
        "K": {"units": "stk64s4, stk32s4, stk128s4, stk64s8 (olmoe-1b-7b, G = 64, n 4, 6, NATIVE, stamps=iter)",
              "readout": "T_iter = median over live CTAs and iterations (the first 2 marks and the last 1 dropped) of consecutive top-of-iteration clock64 differences, per (unit, GEMM, n), median over n",
              "prediction": "T_iter = occ x c_law(BK, stages, occ): in the closed queue each of occ resident CTAs completes a k-step every occ x c",
              "predicted_T_iter_cycles": kpred,
              "classes": classes,
              "classes_basis": "LK and LITTLE predict identical T_iter on all 8 cells (0.00%); MVA2 and OCC differ by at most 3.08% here and by at most 5.14% at any 32x64 w8 BK 64 config of occupancy 2 to 5 (s2 to s8), under two 3% bands (6%), so no lever cell exists and none is added (build-r4-review section 3, work/k_separation.txt)",
              "separates": ["PS from the rest (stk32s4, stk64s8)", "LK=LITTLE from MVA2~OCC (stk64s8 only, 10.8 to 12.8%)"],
              "does_not_separate": ["MVA2 from OCC", "LK from LITTLE"],
              "rule": "a law is consistent on a (unit, GEMM) when |T_meas / T_pred - 1| <= 3% (the stamps' own allowance: the perturb gate bounds the copy at 1% median, 2% worst on the whole kernel); a CLASS is consistent when any of its laws is consistent on every scored (unit, GEMM), FALSIFIED otherwise; a class is SELECTED when it alone is consistent; else UNDECIDED. No single law of MVA2~OCC or LK=LITTLE is ever SELECTED",
              "one_test": "the K block and B' (occlaw) share configurations: ONE test of the k-step law, reported together",
              "labels": {"stk64s4": "SEEN (OLMoE G = 64 BK 64 s4 has a published page, 2026-09-29)", "others": "BLIND-CALMODEL",
                         "in_sample": "MVA2 is in-sample at the k32s4, k64s4 and k64s8 configurations"}},
        "T": {"units": "sttail on mixtral-8x22b (G = 8, n 1, NATIVE and PRIVATE) and on mixtral-8x7b (G = 8, n 1, 2, NATIVE)",
              "readout": "the TAIL WAVE: with N live CTAs and a wave of 132 x occ, the last N - floor(N / wave) x wave live CTAs in start order (r4common.tail_ctas; none when N is whole waves or under one); T_tail = median of their per-iteration clock64 cycles past the first, in ns at 1710 (review_fix: section 3, the earlier rule read waves 2..n)",
              "rivals_ns": tail, "rivals_basis": f"one stage ({STAGE_BYTES} B) per iteration at R1: rho 11.48 GB/s (CORES_RHO, about one stage in flight at loaded latency, REANALYSIS 1c); R3L: three stages in flight at the loaded 1433 ns; R3I: three stages at the idle 331 ns; FS (device share, REANALYSIS 1c): the tail's CTAs share the DRAM ruler, one stage each per iteration, T = stage bytes x N_tail / bw (bw 3598 GB/s, CAL), N_tail read off the launch",
              "FS_registered": fs,
              "occupancy": "w1 5, w2 4 (32x64x64 s4, SEEN cubins)",
              "rule": "r4common.nearest in log space per (unit, GEMM, arm); the reading is SELECTED on a unit when every (GEMM, arm) selects it"},
        "D": {"units": "stdead4 (olmoe-1b-7b, G = 8, n 2, 4, NATIVE and SHARED, s4: occupancy 5 / 4) and stdead2 (num_stages 8: occupancy 2 / 2), stamps=cta",
              "readout_life": "median over dead CTAs (kind 2) of end_clk - start_clk, cycles: the slot a dead CTA holds",
              "rivals_life_cycles": {"FAST": 150.0, "MIX": 451.0, "DRAM": r6(cyc["dram_ns"] + 75)},
              "rivals_life_basis": "FAST: the 75 instructions to EXIT with no exposed load (REANALYSIS fact 4); MIX: L = 264 ns = 451 clk (REANALYSIS 1f, SEEN-fitted on the dead-CTA ratio); DRAM: one DRAM-latency load plus the 75 instructions",
              "readout_dispatch": f"SHARED cells: only the dead CTAs that START AFTER THE LAST LIVE CTA ENDS (the live drain, the kappa term, out): (max - min of their start globaltimer) / (count - 1), ns per dead CTA (r4common.dead_dispatch_ns; review_fix: section 3)",
              "timer_resolution_rule": f"globaltimer's resolution is the least positive difference between the launch's distinct globaltimer values; the dispatch readout is NOT SCORED unless at least {C4.DISPATCH_MIN_CTAS} qualifying dead CTAs span at least {C4.DISPATCH_MIN_TICKS} ticks; the life readout uses clock64 and needs no such rule",
              "rivals_dispatch_ns": disp,
              "rivals_dispatch_basis": "DISP: D's d 0.995 ns (CAL); MIX: d0 + L / (SMs occ) with d0 0.53 ns, L 264 ns (REANALYSIS 1f, SEEN-fitted); SLOT: t / occ, t 4.05 ns (CAL refit)",
              "rule": "r4common.nearest (linear) per (unit, GEMM); the dispatch readout is scored at the GEMM's own occupancy (stdead2 at 2 separates the three by 0.5 ns and more; at 5 and 4 they lie within 0.05 ns of each other and read BETWEEN by design)"},
        "rrze_gh200": RRZE_GH200,
        "missing": "a unit without a PASS gate or with no stamps.json is NOT SCORED; the drop-group st goes together",
        "seen_data": ["every rival's constant is SEEN (CAL, SEEN-fitted or SEEN third-party, per entry)",
                      "no stamp of this kernel exists; the stk64s4 configuration (and occlaw's k64s4 base) is SEEN on OLMoE's 2026-09-29 G = 64 page, the other units are BLIND-CALMODEL or BLIND"],
    }


# --------------------------------------------------------------------------
# hw: gpubench and the rulers
# --------------------------------------------------------------------------

def hw() -> dict:
    import gpubench as GB
    import hw_rulers as HR
    return {
        "registered": f"{C4.DATE}, before any rental-4 page",
        "name": C4.NAMES["hw"],
        "status": "REGISTERED BEFORE ANY PAGE: hardware constants measured independently of any fused_moe page, and consistency checks",
        "design": "owner decision 3 (RRZE-HPC/gpu-benches on the VM, GPL-3.0, fetched at a pinned commit, never vendored) and REANALYSIS.md section 4 (the rulers step)",
        "tool": "scripts/gpubench.py and scripts/hw_rulers.py on the VM; scorer scripts/scoring/rental4/score_hw.py",
        "gpubench": {"pin": GB.PIN, "benches": [b for b, _t, _c in GB.BENCHES],
                     "rules": {"near_l2": f"median ns over buffers from 4 MiB to {GB.LAT_NEAR[1]} L", "far_l2": f"median ns over {GB.LAT_FAR[0]} L to {GB.LAT_FAR[1]} L",
                               "dram": f"median ns over buffers >= {GB.LAT_DRAM} L", "l2_plateau": f"median GB/s over total volume <= {GB.L2C_PLATEAU} L",
                               "l2_floor": f">= {GB.L2C_FLOOR} L", "l2_half_way": "the log-volume interpolation of the first fall below (plateau + floor) / 2",
                               "stream": f"peaks per kernel; the least occupancy reaching {GB.STREAM_KNEE} of the triad peak"},
                     "L": "the card's L2 (60 MiB on GH200; the rulers' device query is printed beside it)"},
        "rulers": {k: HR.plan(4.0)[k]["what"] for k in HR.RULERS},
        "checks": {"far_l2": {"against_ns": RRZE_GH200["far_l2_ns"], "band": 0.05},
                   "dram": {"against_ns": RRZE_GH200["dram_ns"], "band": 0.10},
                   "triad": {"against_gbps": RRZE_GH200["triad_gbps"], "band": 0.03},
                   "l2_cache_size": {"against_bytes": 62914560, "band": 0.0},
                   "rule": "each a CONSISTENT / DIFFERS line against the published GH200 value (SEEN third-party) within its band; latencies are compared in ns (the published run was at 1980 MHz, this one at the 1710 lock: an L2 or DRAM latency fixed in ns reads the same)"},
        "printed": {"eta_mix": f"bw {BW_CAL} GB/s (CAL, the 8x7B fit) over each streaming ruler (read2d, copy, add) and over triad", "cycles_at_lock": "every latency x 1.710",
                    "tau_mma": "matmul TFLOP/s over 132 SMs x 1.710 GHz: FLOP per clock per SM, an upper bound for mma.sync"},
        "use": "the stamps F readout's rivals take this rental's latencies when gpubench ran and parsed",
        "licence": "gpu-benches' code never enters the repository; its printed outputs are measurements published with the session, attributed to RRZE-HPC (GPL-3.0 applies to the code, not to a measurement)",
        "rrze_gh200": RRZE_GH200,
        "seen_data": ["the comparison values are SEEN third-party (RRZE's published GH200 run)", "bw 3598 is CAL"],
    }


# --------------------------------------------------------------------------
# nativegates: owner decision 1, for rental 5's NATIVE-only histogram page
# --------------------------------------------------------------------------

def nativegates() -> dict:
    return {
        "registered": f"{C4.DATE}, before any NATIVE-only histogram page",
        "name": C4.NAMES["nativegates"],
        "status": "REGISTERED GATE DEFINITION (owner decision 1, 2026-10-06); NO RENTAL-4 UNIT USES IT: the histogram page and its plumbing (balanced_ids replaced by realize_counts) are rental 5's",
        "design": "REVIEW.md section 1 (the NATIVE-only, M-ladder, histogram timed page) and section 4 (the uniform-path control, the token shuffle); REANALYSIS.md section 4 owner decision 1",
        "gates": {"G1_lock_thermal": "V7: the 1710 lock in force and the thermal gate, as every timed page",
                  "G2_host_bound": "the host-bound guard (R3's): a cell whose host enqueue exceeds its GPU time is excluded",
                  "G3_uniform_control": "the uniform histogram through the new path matches R3's NATIVE cell (balanced_ids) on the same board within 2 sigma_page (sigma_page 0.18%, SEEN rental 3)",
                  "G4_worst_cell_clock": "the worst cell's measured SM clock is 1710 MHz (no cell below the lock)",
                  "G5_provenance": "the page records the histogram file's sha256 and the token-shuffle seed"},
        "rule": "a NATIVE-only histogram page is VALID only when G1 to G5 all pass; G3 failing voids every histogram cell of its board",
        "seen_data": ["sigma_page is SEEN"],
    }


# --------------------------------------------------------------------------
# rendering, build, check
# --------------------------------------------------------------------------

def _flat(prefix, v, out, depth=0):
    if isinstance(v, dict):
        for k, x in v.items():
            if isinstance(x, (dict, list)) and depth < 2:
                out.append(f"{prefix}{k}:")
                _flat(prefix + "  ", x, out, depth + 1)
            else:
                out.append(f"{prefix}{k}: {x if isinstance(x, str) else json.dumps(x, default=str)}")
    elif isinstance(v, list):
        for x in v:
            out.append(f"{prefix}- {x if isinstance(x, str) else json.dumps(x, default=str)}")
    else:
        out.append(f"{prefix}{v}")


SKIP_TXT = {"code_pins", "laws_on_seen_points"}


def txt(part: str, d: dict) -> list[str]:
    L = [f"REGISTERED {d['registered']}: rental 4, {d['name']}", f"status: {d['status']}",
         f"design: {d['design']}", f"tool: {d.get('tool', '-')}", ""]
    if part == "dead":
        L.append("predicted Delta (us), per contrast:")
        for k, v in d["predictions_us"].items():
            L.append(f"  {k} {v['model']} copies {v['copies']}: " + "  ".join(
                f"{r} {x['total']:.2f}" for r, x in v["rivals"].items()) + f"  sigma_D {v['sigma_D_us']:.2f}")
    if part == "occlaw":
        L.append("predicted ratio to k64s4 (w1 / w2), per law:")
        for page, v in d["predicted_ratio"].items():
            L.append(f"  {page:7s} " + "  ".join(
                f"{law} {v['w1'][law] if v['w1'][law] is None else round(v['w1'][law], 4)}/"
                f"{v['w2'][law] if v['w2'][law] is None else round(v['w2'][law], 4)}" for law in LAWS))
        L.append("  laws on the SEEN points: " + json.dumps(d["laws_on_seen_points"]["rms_by_law"]))
    L.append("")
    for k, v in d.items():
        if k in SKIP_TXT or k in ("registered", "name", "status", "design", "tool"):
            continue
        _flat("", {k: v}, L)
    return L


BUILDERS = {"dead": dead, "occlaw": occlaw, "perturb": perturb, "stamps": stamps,
            "hw": hw, "nativegates": nativegates}


def build() -> dict:
    docs = {p: BUILDERS[p]() for p in C4.PARTS}
    pins = {p: sha256(REPO / p) for p in (
        "scripts/r3_timing_model.py", "scripts/instr_probe.py", "scripts/gpubench.py", "scripts/hw_rulers.py",
        "moe/instrumented/__init__.py", "moe/instrumented/fused_moe_instr.py", "moe/instrumented/cubin.py",
        "scripts/scoring/rental4/r4common.py")}
    for d in docs.values():
        d["code_pins"] = pins
    return docs


def _strip(d: dict) -> dict:
    return {k: v for k, v in d.items() if k != "code_pins"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=REG)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    docs = build()
    bad = 0
    for part, doc in docs.items():
        jpath = a.out_dir / f"{C4.NAMES[part]}.json"
        tpath = a.out_dir / f"{C4.NAMES[part]}.txt"
        jtext = json.dumps(doc, indent=1, default=str) + "\n"
        ttext = "\n".join(txt(part, doc)) + "\n"
        if a.check:
            same_j = jpath.exists() and _strip(json.loads(jpath.read_text())) == _strip(json.loads(jtext))
            same_t = tpath.exists() and tpath.read_text() == ttext
            for p, same in ((jpath, same_j), (tpath, same_t)):
                print(f"{p.name}: {'same' if same else 'DIFFERS'}")
                bad += not same
        else:
            jpath.write_text(jtext)
            tpath.write_text(ttext)
            print(f"wrote {jpath}\nwrote {tpath}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
