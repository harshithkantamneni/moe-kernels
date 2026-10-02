#!/usr/bin/env python
"""Writes rental 2's four registered JSON files (docs/registered/2026-10-01-rental2-*).

    python scripts/scoring/rental2/register.py [--out-dir docs/registered] [--check]

Every number is computed here from committed files: the calibration models'
published pages (CAL), rental 1's published pages and scores where a number is
SEEN (labelled), and the registered 2026-09-30 / 2026-10-01 files. Nothing is
typed in. `--check` recomputes and compares with the committed JSON (exit 1 on
any difference) instead of writing. About a minute on a laptop CPU (the CORES
model is built from the 2026-09-27 pages once).
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as CM  # noqa: E402

REPO = CM.REPO
PUB = REPO / "results" / "published"
R1S = PUB / "2026-10-01-nvidia_gh200_480gb-rental1-session" / "results"
REG = REPO / "docs" / "registered"
TREADS_N = list(range(2, 10))
SCORED_N = list(range(4, 10))
S_NOISE = 0.03          # s noise per page (rental 1's +-0.03 board reproduction of s)
NL_HALF = 0.05          # the no-lag rival's band (rental 1's same-board +-0.05)
EDGE_X = 0.92
C_KSTEP = 344.1          # cycles per CTA k-step (0f77622, CAL counters)
F_CTA = {"w1": 520.0, "w2": 979.0}
SIGMA_Q = 0.03641213879116405   # rental 1 T2's sigma_q (atile registration), CAL


def r6(x):
    return None if x is None else round(float(x), 6)


# --------------------------------------------------------------------------
# part 1: knobs
# --------------------------------------------------------------------------

def knobs() -> dict:
    import wave_split_bytes as W
    l2reg = json.loads((REG / "2026-10-01-l2-survival-tp-gh200.json").read_text())
    atreg = json.loads((REG / "2026-10-01-atile-ksteps-gh200.json").read_text())
    r1 = json.loads((REPO / "scripts/scoring/rental1/l2.score.json").read_text())["s"]
    a, b = r1["tp2 w2"], r1["tp4 w1"]
    d_a, d_b = a["d"], b["d"]
    lam = {}
    for i, n in enumerate(TREADS_N):
        sa, sb = a["meas"][i], b["meas"][i]
        ln = math.log(sa / sb)
        L = (d_b - d_a) / ln
        sig = (d_b - d_a) / ln ** 2 * math.sqrt((S_NOISE / sa) ** 2 + (S_NOISE / sb) ** 2)
        lam[str(n)] = {"lam": r6(L), "sigma": r6(sig), "rel": r6(sig / L),
                       "s_tp2_w2": r6(sa), "s_tp4_w1": r6(sb)}
    params = l2reg["law_params_by_n"]
    steep = {n: params[n]["lam"] for n in params}
    knots = {n: [[0.286, 1.0], [0.576, params[n]["s_cap_0.576"]], [EDGE_X, 0.0]]
             for n in params}
    # the cells, their P, x and d at the default W_c (the scorer re-reads W_c off each page)
    from moe.spec import MODEL_CONFIGS
    import l2_survival as L
    cells = {}
    for model in ("mixtral-8x7b-tp8", "mixtral-8x7b-tp4", "mixtral-8x7b-tp2", "mixtral-8x7b"):
        cfg = MODEL_CONFIGS[model]
        for g, wc in (("w1", 660), ("w2", 528)):
            k, ncols = L.gemm_shape(cfg, g)
            P = ncols // 64
            cells[f"{model} {g}"] = {"P": P, "x": r6((P - 1) * 64 * k * 2 / L.L2_REF_BYTES),
                                     "W_c_default": wc, "d_default": r6(P / wc)}
    # illustrations on rental 1's s_base (SEEN): the bands at the planned W_c
    illus = {}
    meas = {"mixtral-8x7b-tp2 w2": a["meas"], "mixtral-8x7b-tp4 w1": b["meas"],
            "mixtral-8x7b-tp2 w1": r1["tp2 w1"]["meas"], "mixtral-8x7b-tp8 w2": r1["tp8 w2"]["meas"],
            "mixtral-8x7b-tp4 w2": r1["tp4 w2"]["meas"],
            "mixtral-8x7b w2": r1["sameboard 8x7B w2"]["meas"]}
    for key, wc1, knob in (("mixtral-8x7b-tp2 w2", 264, "s8"), ("mixtral-8x7b-tp2 w2", 396, "s6"),
                           ("mixtral-8x7b-tp4 w1", 264, "s8"), ("mixtral-8x7b-tp2 w1", 264, "s8"),
                           ("mixtral-8x7b-tp2 w1", 396, "s6"), ("mixtral-8x7b-tp8 w2", 264, "s8"),
                           ("mixtral-8x7b-tp4 w2", 264, "s8"), ("mixtral-8x7b w2", 264, "s8")):
        c = cells[key]
        dd = c["P"] / wc1 - c["d_default"]
        rows = {}
        for i, n in enumerate(TREADS_N):
            bn = CM.bands(meas[key][i], CM.s_cap(c["x"], knots[str(n)]), dd,
                          lam=lam[str(n)]["lam"], lam_sigma=lam[str(n)]["sigma"],
                          lam_steep=steep[str(n)], past_edge=c["x"] >= EDGE_X,
                          s_noise=S_NOISE, nl_half=NL_HALF)
            rows[str(n)] = {k2: [r6(v[0]), r6(v[1])] for k2, v in bn.items()}
            rows[str(n)]["L2r_central"] = r6(CM.law_value(
                meas[key][i], CM.s_cap(c["x"], knots[str(n)]), dd, lam[str(n)]["lam"],
                c["x"] >= EDGE_X))
        illus[f"{key} {knob} (W_c {wc1})"] = {"dd": r6(dd), "s_base_rental1_SEEN":
                                               [r6(v) for v in meas[key]], "bands": rows}
    # ---- T5: candidates on the page's geometry; registered centres at the default W_c
    fits = atreg["fits"]["primary (calibration-only: 8x7B 09-27, 8x22B, Qwen2-57B, OLMoE)"]
    cand = {"R0": {"params": l2reg["secondary_G2"]["R0_params"], "content_a": False,
                   "label": "CAL: the registered 8x7B MIX fit (R0 of the 2026-10-01 registrations)"},
            "R1": {"params": fits["R1"]["params"], "content_a": True,
                   "label": "CAL: 2026-10-01 atile registration, primary R1"},
            "R2": {"params": fits["R2"]["params"], "content_a": True,
                   "label": "CAL: 2026-10-01 atile registration, primary R2"}}
    tp2_page = json.loads((R1S / "2026-10-01-nvidia_gh200_480gb-mixtral-8x7b-tp2-atile-r3-counters"
                           / "lock1710" / "r3c-g64.json").read_text())
    sb_page = json.loads((R1S / "2026-10-01-nvidia_gh200_480gb-mixtral-8x7b-sameboard-r3-counters"
                          / "lock1710" / "r3c-g32.json").read_text())
    g_tp2, g_87 = W.page_geometry(tp2_page), W.page_geometry(sb_page)
    p2 = W.Params.from_json(fits["R2"]["params"])

    def rho84(m):
        pp = dataclasses.replace(p2, a_A=p2.a_A * m)
        return (CM.price_f(g_tp2, pp, True, G=64) / CM.price_f(g_87, pp, True, G=32))
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if rho84(mid) < 1.184:
            lo = mid
        else:
            hi = mid
    m = (lo + hi) / 2
    p2h = dataclasses.replace(p2, a_A=p2.a_A * m)
    cand["R2h"] = {"params": json.loads(json.dumps(dataclasses.asdict(p2h))), "content_a": True,
                   "m": r6(m), "label": "SEEN: R2 with a_A x m, m fitted to rental 1's measured "
                                        "rho84 = 1.184 alone"}
    prm = CM.t5_candidates({"candidates": cand})
    centres = {}
    for name, (pp, ca) in prm.items():
        f = {bk: CM.price_f(dataclasses.replace(g_tp2, block_k=bk), pp, ca) for bk in (32, 64, 128)}
        rho = f[128] / f[32]
        s = CM.sigma_rho(rho, f[128], f[32], SIGMA_Q)
        centres[name] = {"f9": {str(k): r6(v) for k, v in f.items()}, "rho_BK": r6(rho),
                         "sigma_rho": r6(s), "half_width": r6(2 * s),
                         "band_at_default_Wc": [r6(rho - 2 * s), r6(rho + 2 * s)]}
    s8 = {}
    for name, (pp, ca) in prm.items():
        g8 = dataclasses.replace(g_tp2, W_c=(("w1", 264), ("w2", 264)),
                                 ctas_per_sm=(("w1", 2), ("w2", 2)))
        s8[name] = r6(CM.price_f(g8, pp, ca))
    return {
        "registered": f"{CM.DATE}, before any rental-2 page",
        "name": CM.NAMES["knobs"],
        "status": "REGISTERED BEFORE ANY PAGE: no rental-2 page exists when this is committed",
        "design": "scratchpad design-r2/DESIGN.md section 1 with design-r2-review/REVIEW.md sections 1 and 5.1",
        "tool": "scripts/scoring/rental2/register.py (this file's numbers); scorer scripts/scoring/rental2/score_knobs.py",
        "measure": {
            "s": "scripts/l2_survival.py: s(n) = (B_PRIVATE - B_SHARED) / ((n - 1) W_g), G = 1, n = 2..9, B = per_gemm[g].dram_bytes_read",
            "P": "npn = N / BLOCK_N (w1 N = 2F, w2 N = H)",
            "x": "(P - 1) BLOCK_N K b / 60 MiB",
            "W_c": "page's sm_count x the least of the four recorded occupancy limits, per GEMM (wave_split_bytes.page_window), read off EACH page, never assumed from the knob",
            "d": "P / W_c",
            "dd": "d(knob page) - d(same-board base page), each page's own W_c",
        },
        "law_L2r": {
            "form": "past the edge (x >= 0.92): s_knob = s_base exp(-dd / lam(n)); below it: s_knob = s_cap(x, n) + (s_base - s_cap(x, n)) exp(-dd / lam(n)); r(lam) = exp(-dd / lam)",
            "lambda_by_n_SEEN": lam,
            "lambda_source": "rental 1's same-x pair, tp2 w2 (d %.4f) and tp4 w1 (d %.4f): lam = (d_b - d_a) / ln(s_a / s_b), SEEN (scripts/scoring/rental1/l2.score.json)" % (d_a, d_b),
            "lambda_band": "per n, lam +- sigma, sigma the +-0.03 noise of each s propagated in quadrature (lambda_by_n_SEEN.rel: 42% at n = 4, 33% at n = 5, 24 to 34% at n = 6..9; the review estimated 46% and 35%); n = 2, 3 printed only",
            "s_cap_knots_by_n": knots,
            "s_cap_source": "1 at x <= 0.286 (CAL: Qwen2 w2, OLMoE); the 0.576 value per n solved from Qwen2 w1 (CAL, the 2026-10-01 L2 registration's law_params_by_n); 0 at x >= 0.92 (SEEN: rental 1's tp2 w2 sits on 8x7B w2's curve); linear between knots",
        },
        "rivals": {
            "NL": "no lag: s_knob = s_base (H0, H1 or any pure-capacity law); band s_base +- 0.05",
            "ST": "steep: the registered 2026-10-01 lam(n) (law_params_by_n, CAL but x-confounded) in the same law form; band [0, prediction + 0.03]",
            "lambda_steep_by_n": steep,
            "H2c": "k-steps: s moves with BLOCK_K at fixed W_c",
            "H3": "address: s moves with slot padding",
        },
        "bands": {"L2r": "[min(pred(lam - sigma), pred(lam + sigma)) - 0.03, max(...) + 0.03], pred at lam - sigma <= 0 is the infinitely steep value",
                  "NL": "[s_base - 0.05, s_base + 0.05]", "ST": "[0, pred(lam_steep) + 0.03]",
                  "s_noise": S_NOISE, "nl_half": NL_HALF},
        "cells": cells,
        "pages": {
            "path": "<date>-nvidia_gh200_480gb-<model>-<label>-r3-counters/lock1710/r3c-g<G>.json under the run tree",
            "base": {"mixtral-8x7b-tp2": "l2base", "mixtral-8x7b-tp4": "l2base", "mixtral-8x7b": "l2base"},
            "knob": {"l2s8": {"num_stages": 8}, "l2s6": {"num_stages": 6},
                     "l2bk32": {"block_k": 32, "num_stages": 4}, "l2bk128": {"block_k": 128, "num_stages": 2},
                     "l2pad7": {"slot_pad_rows": 7}},
            "G": 1,
        },
        "tests": {
            "tp2 w2 s8 (PRIMARY)": {"page": ["mixtral-8x7b-tp2", "l2s8"], "base": "l2base", "gemm": "w2",
                                    "hypotheses": ["L2r", "NL", "ST"], "n": SCORED_N, "need": 4,
                                    "rule": "a hypothesis is SELECTED when 4 of n = 4..9 fall in its band and in no other band; FALSIFIED when 4 of 6 fall outside its band; else INCONCLUSIVE. The L2r-vs-ST call is made here."},
            "tp4 w1 s8 (PRIMARY, NL vs lag)": {"page": ["mixtral-8x7b-tp4", "l2s8"], "base": "l2base", "gemm": "w1",
                                              "hypotheses": ["LAG", "NL"], "n": SCORED_N, "need": 4,
                                              "rule": "NL against LAG, LAG's band the envelope of L2r and ST (ST's band lies inside L2r's here at dd 0.254, so the two are not told apart: the review, section 1)"},
            "tp2 w2 s6 (secondary, the curve's shape)": {"page": ["mixtral-8x7b-tp2", "l2s6"], "base": "l2base", "gemm": "w2",
                                                         "hypotheses": ["L2r", "NL", "ST"], "n": [6, 7, 8, 9], "need": 4,
                                                         "rule": "SELECTED needs all four of n = 6..9 in its band and in no other (L2r's upper edge sits 0.006 to 0.05 under NL's lower edge there)"},
            "8x7B w2 s8 (x-invariance, tail)": {"page": ["mixtral-8x7b", "l2s8"], "against": ["mixtral-8x7b-tp2", "l2s8"], "gemm": "w2",
                                               "n": SCORED_N, "rule": "HELD when |s(8x7B w2 s8) - s(tp2 w2 s8)| <= 0.05 at 4 or more of n = 4..9, else FALSIFIED (an x-law puts 8x7B lower)"},
            "tp4 w2 s8": {"page": ["mixtral-8x7b-tp4", "l2s8"], "base": "l2base", "gemm": "w2", "rule": "RECORD: below the edge L2r moves it by 0.05 to 0.08, about the noise"},
            "H2c BLOCK_K at G=1 (tail)": {"pages": [["mixtral-8x7b-tp2", "l2bk32"], ["mixtral-8x7b-tp2", "l2bk128"]], "base": "l2base", "gemms": ["w1", "w2"],
                                          "rule": "per page and GEMM, centre = s_base when the page's W_c equals the base's, else the L2r central value at the page's dd; NO k-step effect (H2c FALSIFIED) when |s - centre| <= 0.05 at 4 or more of n = 4..9; H2c SELECTED when |s - centre| > 0.05 at 4 or more; else INCONCLUSIVE"},
            "H3 slot pad 7 (tail)": {"page": ["mixtral-8x7b-tp2", "l2pad7"], "base": "l2base",
                                     "rule": "H3 FALSIFIED (no address effect) when every PRIVATE per-GEMM dram_bytes_read is within 0.3% of the base page's AND |s - s_base| <= 0.03 at 4 or more of n = 4..9 on both GEMMs; H3 SELECTED when |s - s_base| > 0.03 at 4 or more on a GEMM or a PRIVATE byte count moves by more than 1%; else INCONCLUSIVE"},
        },
        "controls": {
            "F_over_Mn": {"rule": "every PRIVATE cell of every knob and base page: F / Mn in [0.50, 0.56]", "band": [0.50, 0.56], "scope": "page", "effect": "a failing page's SELECTED verdicts are demoted to INCONCLUSIVE"},
            "tp8 w2 s8 negative control": {"page": ["mixtral-8x7b-tp8", "l2s8"], "gemm": "w2", "rule": "s >= 0.95 at every n = 4..9 (below every edge: L2r's s_cap is 1 at x 0.23)", "scope": "global", "effect": "a failure demotes EVERY part-1 SELECTED verdict to INCONCLUSIVE"},
            "tp2 w1 on s6 and s8": {"rule": "tp2 w1 (x 1.86, past the edge) at or under the upper edge of its own L2r band at every n = 4..9 (derived from L2r, replacing the fixed 0.08 rental 1 failed at n = 2, 3)", "scope": "page", "effect": "a failure demotes that page's L2r (or LAG) SELECTED verdict to INCONCLUSIVE; an NL or ST selection stands, since the control is L2r's own prediction"},
            "V6": {"rule": "a page whose V6 gate is not PASS is FLAGGED: its cells are excluded from every count and its verdicts read INCONCLUSIVE (V6 checks the activation-cancels premise of s)", "scope": "page"},
            "V10": "does not gate any score here (bytes are clock-free)",
        },
        "partition_cross_check": {
            "pages": [["mixtral-8x7b-tp2", "l2base"], ["mixtral-8x7b-tp2", "l2s8"]],
            "rule": "per SHARED and PRIVATE cell n = 2..9 and GEMM: direct fabric read hits (lts__t_sectors_srcunit_ltcfabric_op_read_lookup_hit) / (F - (M - Mn)) within 1 +- 0.05",
            "effect": "an INSTRUMENT check only: no verdict of part 1 reads the far share. When more than 10% of a page's cells fail, the far-share RECORD of that page is printed from the direct count only and the subtraction is flagged",
            "record": "demote counts and the far share, both ways, per page",
        },
        "T5": {
            "what": "C's separator: rho_BK = f(BK128 / s2) / f(BK32 / s4), PRIVATE w2, tp2 G = 64, n = 9 (n = 8 printed), one board; k-steps 56 against 224 at the same L, bytes and W_c",
            "pages": {"BK64": ["mixtral-8x7b-tp2", "atbk64"], "BK32": ["mixtral-8x7b-tp2", "atbk32"], "BK128": ["mixtral-8x7b-tp2", "atbk128"], "s8 RECORD": ["mixtral-8x7b-tp2", "ats8"]},
            "G": 64, "n": 9,
            "f": "f = (q - n) 128 / (63 n), q = dram_counter_route.r3_q(page)['private']['w2'][n], the q wave_split_bytes.load_card(page).measured holds",
            "candidates": cand,
            "registered_at_default_Wc": centres,
            "pricing": "each candidate's f on each page's OWN geometry (wave_split_bytes.page_geometry: its block_k and its W_c), W.Model(geom, content_a).evaluate(params, [('private', 64, 9, 'w2')]); rho = f(BK128 page) / f(BK32 page)",
            "band": "the re-priced centre +- the half-width registered here at the default W_c (2 sigma_rho, sigma_rho from sigma_q %.4f through sigma_f = sigma_q 128 / (63 x 9) on both f in quadrature); the widths stay FIXED when the centres are re-priced" % SIGMA_Q,
            "rule": "a candidate is FALSIFIED when the measured rho_BK lies outside its band, HELD inside; the verdict is on n = 9 alone",
            "board_check": {"page": "BK64", "f_rental1": 0.5802, "tolerance_rel": 0.028, "rule": "f(BK64, n = 9) within 2.8% of rental 1's 0.5802 (the 09-25 / 09-27 board-to-board PRIVATE w2 reproduction)", "effect": "a failure turns R2h's verdict (its m is rental 1's board's) to INCONCLUSIVE; R0, R1 and R2 stand"},
            "s8_record_at_Wc264": s8,
            "labels": {"R0": "CAL", "R1": "CAL", "R2": "CAL", "R2h": "SEEN"},
        },
        "illustrations_on_rental1_SEEN_bases": illus,
        "seen_data": [
            "lam(n) is fitted on rental 1's tp2 w2 and tp4 w1 G = 1 pages (SEEN)",
            "s_cap = 0 at x >= 0.92 is read off rental 1's tp2 w2 (SEEN)",
            "R2h's m is fitted on rental 1's rho84 = 1.184 (SEEN)",
            "the illustrations use rental 1's measured s as s_base; the scores use each knob page's same-board base page",
        ],
    }


# --------------------------------------------------------------------------
# part 2: launch2
# --------------------------------------------------------------------------

def launch2() -> dict:
    import cores_heldout_predict as CP
    import cross_model_predict as X
    lf1 = json.loads((REG / "2026-10-01-launch-floor-gh200.json").read_text())
    d = X.predict(CP.source_pages(), CP.C27, "mixtral-8x7b-tp8", False)
    creg = {}
    for c in d["cells"]:
        if c["G"] == 4 and c["n"] <= 9:
            creg.setdefault(c["arm"], {})[str(c["n"])] = round(float(c["ms"]), 4)
    granite = lf1["C_reg_ms"]["granite-3.0-3b-a800m"]
    f240 = 251.7e6 / 3.7261e12 * 1e3
    f480 = 480 * 2 ** 20 / 3.7261e12 * 1e3
    return {
        "registered": f"{CM.DATE}, before any rental-2 page",
        "name": CM.NAMES["launch2"],
        "status": "REGISTERED BEFORE ANY PAGE",
        "design": "design-r2/DESIGN.md section 2 with REVIEW.md sections 2 and 5.2",
        "tool": "scripts/launch_floor.py --phase timed, then --phase trace (two processes, one lock); scorer scripts/scoring/rental2/score_launch.py",
        "models": {"mixtral-8x7b-tp8": "BLIND on every host-side prediction (no timed page of it exists); P1's GR is C_reg CAL with its GEMM durations SEEN under ncu (rental-1 r3c gpu_time_ns at G = 1, 2, 8)",
                   "granite-3.0-3b-a800m": "SEEN: the rerun of rental 1's confounded P2, P4 and P8",
                   "jetmoe-8b": "tail unit, scored by the same rules", "granite-3.0-1b-a400m": "tail unit, scored by the same rules"},
        "C_reg_ms": {"mixtral-8x7b-tp8": creg, "granite-3.0-3b-a800m": granite,
                     "jetmoe-8b": lf1["C_reg_ms"]["jetmoe-8b"],
                     "granite-3.0-1b-a400m": lf1["C_reg_ms"]["granite-3.0-1b-a400m"],
                     "source": {"mixtral-8x7b-tp8": "scripts/cross_model_predict.py predict() at this commit, 8x7B 2026-09-27 source pages (cores_heldout_predict.source_pages, counters C27), G = 4 (CAL)",
                                "others": "the 2026-10-01 launch-floor registration's C_reg_ms, unchanged"}},
        "F_ms": {"E240": round(f240, 4), "E480": round(f480, 4), "E0": 0.0,
                 "basis": "bytes over the 3.7261 TB/s ruler: 251.7 MB (E240) and 480 MiB (E480)",
                 "F_tr": "the flush kernel's duration in the TRACE process (TR-E median flush_us)"},
        "definitions": {
            "I": "a cell's median over its repeats of ms_p50",
            "H_cell": "the median over the cell's repeats of host_enqueue_ms / calls_per_burst (cells.csv)",
            "H_pre": "hostprobe.csv H_pre_ms of the cell's (arm, n, mode): its own host time measured in the same process before any GPU cell",
            "host_bound_fraction": "the fraction of the cell's repeat rows (3) whose host_bound is True",
            "measured_host_bound": "host_bound_fraction >= 0.8; measured GPU-bound: host_bound_fraction == 0",
            "rule_host_bound": "C_reg + F_mode < H_pre (F_mode the cell's own mode's F)",
            "edge": "|C_reg + F_mode - H_pre| <= 0.020 ms (by rule), or host_bound_fraction in [0.2, 0.8] (measured); edge cells are out of P2, P4, P5 and P8",
        },
        "P0": {"what": "an instrument gate", "fails_if": [
            "the timed process's manifest is not phase 'timed', has profiler_enabled_ever True, or made no profiler check",
            "the median over (arm, n, mode) of |H_post / H_pre - 1| exceeds 0.05",
            "H_cell / H_pre lies outside [0.93, 1.10] on more than 5% of the probed cells",
            "more than 5% of the probe rows are REFUSED (a synchronising body or a GPU not held)"],
            "effect": "P2, P4, P5 and P8 are still printed, marked 'host drift'"},
        "P1": {"cells": "GR: tp8 every (arm, n <= 9), 27 cells; Granite n <= 5, 15 cells",
               "band": [0.80, 1.10], "rule": "GR in [0.80, 1.10] x C_reg; FALSIFIED when more than 1 cell per model lies outside, or an increment is short: Granite I_GR(5) - I_GR(1) >= 0.10 ms for SHARED and PRIVATE; tp8 I_GR(9) - I_GR(1) >= 0.8 x (C_reg(9) - C_reg(1)) per arm",
               "labels": {"mixtral-8x7b-tp8": "C_reg CAL; GEMM durations SEEN under ncu (rental-1 r3c gpu_time_ns, G = 1, 2, 8)", "granite-3.0-3b-a800m": "SEEN replication"}},
        "P2": {"cells": "E240 cells GPU-bound by rule with margin (C_reg + F240 > H_pre + 0.020 ms) AND measured GPU-bound (host_bound_fraction 0)",
               "rule": "0 <= E240 - GR <= max(2% of E240, 9 us); FALSIFIED when more than 1 per model lies outside", "label": "band SEEN (rental 1: 3.8 to 6.0 us on 13 cells)"},
        "P3": {"cells": "n = 1, 2, every arm, both models: measured host-bound and non-edge at both E0 and E240, with both TR-E E240 and TR-G E0/E240 traces",
               "rule": "E0 - E240 in [F_tr - h_flush + dD - 12 us, F_tr - h_flush + 12 us], h_flush = hostprobe.csv h_flush_ms of the E240 row (the PROBE's, not the trace's), F_tr = TR-E E240 median flush_us, dD = min(0, TR-G(E0) kernel_sum - TR-G(E240) kernel_sum); FALSIFIED when more than 1 qualifying cell lies outside; NOT TESTED when fewer than 8 qualify"},
        "P4": {"cells": "measured host-bound and non-edge at both E240 and E480",
               "rule": "the shift (E480 - E240) - (H_cell,480 - H_cell,240) = -(F480_tr - F240_tr) +- 12 us, F_tr from TR-E traces (median over the n = 1, 2 cells of both); the E480 verdict by rule (C_reg + F480 < H_pre at E480) against measured on non-edge E480 cells; FALSIFIED when a shift lies outside the band on more than 1 cell, or the verdict is wrong on more than 2 cells per model"},
        "P5": {"cells": "every non-edge E240, E0 and E480 cell", "rule": "measured host-bound iff C_reg + F_mode < H_pre; FALSIFIED under 95% agreement per model"},
        "P6": {"cells": "E240, E0 and E480 at the trace treads, each with its own mode's TR-E and TR-G",
               "rule": "MP within max(5%, 12 us) of I; FALSIFIED when more than 1 per model lies outside",
               "function": "scripts/scoring/rental2/launch_mp.py: timeline() and mp() copied verbatim from scripts/scoring/rental1/score_launch.py (its sha256 is pinned in tests/test_scoring_rental2.py); mp(te, tg, H_us) scales the trace's host offsets by H_us / P_traced, and H_us here is H_cell (rental 1's text already did this)"},
        "P7": {"cells": "tp8 and Granite n = 1", "rule": "classifies only: TR-G kernel sum, non-GEMM time, launch-API share"},
        "P8": {"cells": "non-edge E240 cells host-bound by rule", "rule": "plateau I_E240 in [H_cell - 0.12, H_cell - 0.08] ms; FALSIFIED when more than 10% lie outside per model",
               "label": "Phi band SEEN (Granite); tp8 BLIND"},
        "predicted_host_bound_tp8": {
            "note": "printed estimates at two H; the scores use each cell's own H_pre",
            "H_0.30": {m: {arm: [n for n in range(1, 10) if creg[arm][str(n)] + f < 0.30] for arm in creg}
                       for m, f in (("E240", f240), ("E480", f480), ("E0", 0.0))},
            "H_0.35": {m: {arm: [n for n in range(1, 10) if creg[arm][str(n)] + f < 0.35] for arm in creg}
                       for m, f in (("E240", f240), ("E480", f480), ("E0", 0.0))}},
        "seen_data": ["Granite's P1 and P2 bands and the Phi plateau are SEEN (rental 1 and the 09-30 pages)",
                      "tp8's GEMM gpu_time_ns at G = 1, 2, 8 is on rental-1 r3c pages (SEEN); its host and graph timing are not"],
    }


# --------------------------------------------------------------------------
# parts 3 and 4: the floor captures
# --------------------------------------------------------------------------

CAL_MODELS = {"mixtral-8x7b", "mixtral-8x22b", "qwen2-57b-a14b", "olmoe-1b-7b"}
#: the shallowest tread on no published page, per shard (the review, section 4)
UNSEEN_MIN = {"mixtral-8x7b-tp8": 11, "mixtral-8x7b-tp4": 10, "mixtral-8x7b-tp2": 10}


def _cal_series(CP, TM, params, kw):
    """Per published G=64 floor capture (base and 1710 lock) and GEMM, the
    floor-bound (CORES floor share >= 0.97) cells of >= 2 waves."""
    out = {}
    for f in sorted(PUB.rglob("r3f-g64*.json")):
        if "unlocked" in f.stem:
            continue
        d = json.loads(f.read_text())
        model = d["plan"]["model"]
        treads = sorted({c["n"] for c in d["cells"]})
        pr = CP.predict(model, treads, [64], params, kw)
        P = {(r["n"], r["gemm"]): r for r in pr["cells"]}
        TM.set_model(model)
        clk = "lock" if "lock" in f.stem else "base"
        sess = f.parent.parent.parent.name
        for c in d["cells"]:
            for g in ("w1", "w2"):
                r = P[(c["n"], g)]
                m = c["per_gemm"][g]
                u = CM.u_cycles(TM.GEOMETRY[g].ksteps, g, C_KSTEP, F_CTA)
                q = r["live_ctas"] / 132
                ff = math.ceil(q) * u / r["cores"]
                out.setdefault((model, sess, clk, g), []).append(dict(
                    n=c["n"], q=q, ceil=math.ceil(q), u=u, waves=r["waves"], ff=ff,
                    ac=m["sm__cycles_active.avg"], el=m["sm__cycles_elapsed.avg"],
                    mhz=m.get("sm_clock_mhz"), cores=r["cores"],
                    grid=r["live_ctas"] + r["dead_ctas"], ksteps=TM.GEOMETRY[g].ksteps))
    return out


def const_and_w1floor() -> tuple[dict, dict]:
    import numpy as np
    import cores_heldout_predict as CP
    import r3_timing_model as TM
    import floor_estimator as FE
    src = TM.build(TM.build_parser().parse_args([*map(str, CP.source_pages()), "--counters", str(CP.C27)]))
    params, kw = src["main"].params, src["main"].k_w
    ser = _cal_series(CP, TM, params, kw)
    # ---- part 3, CAL: Z by intercept, per series; gamma base -> lock; the active noise
    rows, sig_active = [], []
    for (model, sess, clk, g), cs in sorted(ser.items()):
        use = [c for c in cs if c["ff"] >= 0.97 and c["waves"] >= 2]
        if len(use) < 3:
            continue
        Z, slope = FE.intercept([c["q"] for c in use], [c["ac"] for c in use])
        res = [c["ac"] - (Z + slope * c["q"]) for c in use]
        sa = math.sqrt(sum(r * r for r in res) / (len(use) - 2)) if len(use) > 2 else None
        rows.append(dict(model=model, session=sess, clock=clk, gemm=g, cells=len(use),
                         u=r6(use[0]["u"]), Z=r6(Z), Zu=r6(Z / use[0]["u"]),
                         slope_over_u=r6(slope / use[0]["u"]),
                         mhz=r6(st.median([c["mhz"] for c in use if c["mhz"]])),
                         active_rms=r6(sa), cal=model in CAL_MODELS))
        if model in CAL_MODELS and sa is not None:
            sig_active.append(sa)
    gam = []
    for r in rows:
        if r["clock"] != "base" or not r["cal"]:
            continue
        l = next((x for x in rows if x["clock"] == "lock" and x["model"] == r["model"]
                  and x["session"] == r["session"] and x["gemm"] == r["gemm"]), None)
        if l and r["Z"] > 0 and l["Z"] > 0:
            gam.append(dict(model=r["model"], session=r["session"], gemm=r["gemm"],
                            gamma=r6(math.log(l["Z"] / r["Z"]) / math.log(l["mhz"] / r["mhz"]))))
    cal_lock_w2 = [r["Zu"] for r in rows if r["cal"] and r["clock"] == "lock" and r["gemm"] == "w2"]
    cal_lock_all = [r["Zu"] for r in rows if r["cal"] and r["clock"] == "lock"]
    sigma_active = r6(st.median(sig_active))
    # ---- part 4, CAL noise: elapsed - CORES about a free constant and slope, >= 4 waves
    sig_cell = []
    for (model, sess, clk, g), cs in sorted(ser.items()):
        if model not in CAL_MODELS:
            continue
        use = [c for c in cs if c["ff"] >= 0.97 and c["waves"] >= 4]
        if len(use) < 3:
            continue
        x = np.array([FE.estimator_x(c["grid"], c["ksteps"]) for c in use])
        y = np.array([c["el"] - c["cores"] for c in use])
        p = np.polyfit(x, y, 1)
        res = y - np.polyval(p, x)
        if len(use) > 2:
            sig_cell.append(math.sqrt(float((res ** 2).sum()) / (len(use) - 2)))
    sigma_cell = r6(st.median(sig_cell))
    # ---- the new captures' cells under the CAL model
    CAPS = {"mixtral-8x7b-tp8": [1, 2, 6, 8, 10, 11, 12, 13, 14, 15, 16],
            "mixtral-8x7b-tp4": [1, 2, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16],
            "mixtral-8x7b-tp2": [1, 2, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16]}
    T1005 = [2, 4, 6, 8, 12, 16]
    cells, geo = {}, {}
    for model, treads in CAPS.items():
        pr = CP.predict(model, treads, [64], params, kw)
        TM.set_model(model)
        geo[model] = {g: dict(pr["geometry"][g], asymptote=pr["floor_per_cta_kstep_cycles"][g],
                              u=r6(CM.u_cycles(pr["geometry"][g]["ksteps"], g, C_KSTEP, F_CTA)))
                      for g in ("w1", "w2")}
        for c in pr["cells"]:
            g = c["gemm"]
            u = geo[model][g]["u"]
            q = c["live_ctas"] / 132
            cells.setdefault(model, []).append(dict(
                n=c["n"], gemm=g, cores=c["cores"], waves=c["waves"], live=c["live_ctas"],
                dead=c["dead_ctas"], grid=c["live_ctas"] + c["dead_ctas"], q=r6(q),
                ceil=math.ceil(q), u=u, floor_share_sigma1=r6(math.ceil(q) * u / c["cores"]),
                floor_bound=bool(math.ceil(q) * u / c["cores"] >= 0.97 and c["waves"] >= 2)))
    # ---- part 3's prediction for K3 under the two forms, and its noise
    k3 = {}
    for g in ("w1", "w2"):
        fb = [c for c in cells["mixtral-8x7b-tp4"] if c["gemm"] == g and c["floor_bound"]]
        q1005 = [c["q"] for c in fb if c["n"] in T1005]
        q1710 = [c["q"] for c in fb]
        u = geo["mixtral-8x7b-tp4"][g]["u"]
        z0 = st.median(cal_lock_w2 if g == "w2" else cal_lock_all) * u
        s1005 = FE.intercept_sigma(q1005, sigma_active) if len(q1005) >= 3 else None
        s1710 = FE.intercept_sigma(q1710, sigma_active)
        rel = math.sqrt((s1005 / z0) ** 2 + (s1710 / z0) ** 2) if s1005 else None
        k3[g] = {"cells_1005": [c["n"] for c in fb if c["n"] in T1005],
                 "cells_1710": [c["n"] for c in fb], "Z_assumed": r6(z0),
                 "sigma_Z_1005": r6(s1005), "sigma_Z_1710": r6(s1710), "sigma_ratio_rel": r6(rel)}
    const = {
        "registered": f"{CM.DATE}, before any rental-2 page",
        "name": CM.NAMES["const"],
        "status": "REGISTERED BEFORE ANY PAGE",
        "design": "design-r2/DESIGN.md section 3 with REVIEW.md sections 3 and 5.3",
        "tool": "scripts/scoring/rental2/score_const.py; the floor captures take --floor-shape-metrics and --floor-null-kernel",
        "captures": {"mixtral-8x7b-tp8": {"label": "floor2", "treads": CAPS["mixtral-8x7b-tp8"], "clocks": ["base", "lock1710"]},
                     "mixtral-8x7b-tp4": {"label": "floor", "treads": CAPS["mixtral-8x7b-tp4"], "clocks": ["base", "lock1710"]},
                     "mixtral-8x7b-tp2": {"label": "floor", "treads": CAPS["mixtral-8x7b-tp2"], "clocks": ["base", "lock1710"]},
                     "mixtral-8x7b-tp4 at 1005": {"label": "floor1005", "treads": T1005, "clocks": ["lock1005"]},
                     "path": "<date>-nvidia_gh200_480gb-<model>-<label>-r3-counters/r3f-g64[-lock<F>].json",
                     "primary": "the 1710 lock captures (part 3 reads clocks, the base capture is printed beside)"},
        "constants": {"c_cycles_per_kstep": C_KSTEP, "F_cycles": F_CTA, "sms": 132,
                      "u": "u = S c + F_g (CAL, 0f77622)", "null_kernel_cycles": 1000},
        "definitions": {
            "q": "N_live / 132, N_live = E n npn (every NATIVE M-tile full)",
            "floor_bound": "the cell's CORES floor share ceil(q) u / CORES >= 0.97 at sigma 1 (the launch order's own reads), NOT re-priced, and waves >= 2; the registered list below is the cell set",
            "Z_f": "the INTERCEPT of sm__cycles_active.avg regressed on q over a capture's floor-bound cells of one GEMM, the slope free (floor_estimator.intercept); one Z per (capture, GEMM)",
            "T": "elapsed.avg - active.avg - (ceil(q) - q) u, per cell (RECORD)",
            "L": "sm__cycles_elapsed.avg - sm__cycles_active.max, per cell",
            "D_imb": "active.max - active.avg - (ceil(q) - q) u, per cell",
            "cta_excess": "sm__ctas_launched.max - ceil(grid / 132), per cell (RECORD: dead CTAs exit at once and are counted)",
            "concurrency": "sm__ctas_active.sum / (132 x sm__cycles_active.avg), per cell (RECORD)",
            "L0": "the null kernel's sm__cycles_elapsed.avg - 1000: a one-CTA launch-and-drain floor, a LOWER BOUND on a 132-SM grid's (RECORD)",
            "clock": "each capture's median over its floor-bound cells of sm_clock_mhz (elapsed / gpu__time_duration), never the lock asked",
        },
        "cells_registered": cells,
        "cal": {"series": rows, "gamma_base_to_lock": gam,
                "Zu_lock_w2": cal_lock_w2, "Zu_lock_all": cal_lock_all,
                "sigma_active_cycles": sigma_active,
                "note": "re-derived with the intercept estimator (the review, section 3); CAL = the four calibration models' G = 64 base and 1710 captures (8x7B 09-27 and 09-30, 8x22B, Qwen2-57B, OLMoE), floor-bound cells of >= 2 waves, >= 3 per series; the other models are printed as seen"},
        "K1": {"pool": "every floor-bound cell of the tp8, tp4 and tp2 1710 captures, both GEMMs", "threshold_cycles": 5000,
               "rule": "HOLDS when L <= 5000 cycles on at least 80% of the pool; FALSIFIED (H_LD carries a material part) when L > 5000 on more than 20%",
               "basis": "CAL T at the lock <= 4.2k cycles, and L <= T"},
        "K2": {"cells": "Z_f / u of w2 on the tp4 (S 56) and tp2 (S 112) 1710 captures, pooled: the median of the two",
               "band": [round(min(cal_lock_w2) - 0.005, 2), round(max(cal_lock_w2) + 0.005, 2)],
               "centre": r6(st.median(cal_lock_w2)),
               "rule": "FALSIFIED when the pooled value lies outside the band; H_LD predicts Z about 0",
               "basis": "the full CAL range of w2 Z_f / u at the 1710 lock, Qwen2 included (no model is excluded)"},
        "K3": {"what": "Z_f(1005) / Z_f(1710) on tp4 w1 and w2, the intercepts over the floor-bound cells the 1005 capture holds (n = 2, 4, 6, 8, 12, 16) and over the 1710 capture's own",
               "forms": {"ns": "a constant in nanoseconds: the ratio equals f_1005 / f_1710 (each capture's own clock)",
                         "cycles": "a constant in SM cycles: the ratio is 1.0"},
               "noise": k3,
               "rule": "per GEMM r = Z_1005 / Z_1710 and rho_f = f_1005 / f_1710; the pooled reading is the mean of the two GEMMs. The ns form HOLDS when |r - rho_f| <= 2 sigma_r and the cycle form is then FALSIFIED if |r - 1| > 2 sigma_r; the cycle form HOLDS when |r - 1| <= 2 sigma_r and the ns form is then FALSIFIED if |r - rho_f| > 2 sigma_r; sigma_r = r x sigma_ratio_rel (the registered noise model). A reading inside both bands, or neither, is INCONCLUSIVE (the zone the review names between the two)",
               "cal_gamma_record": "the CAL base-to-lock exponent is not used as a band: by intercepts it reads %s over %d series, which contains both forms (a 1.26x clock ratio cannot pin it); printed only" % (
                   [g["gamma"] for g in gam], len(gam))},
        "K4": {"rule": "HOLDS when D_imb lies in [-0.2 u, 0.2 u] on at least 80% of the K1 pool; FALSIFIED (H_IMB) when D_imb >= 0.3 u on more than 20%",
               "cta_clause": "RECORD only: sm__ctas_launched.max - ceil(grid / 132) is printed, never scored (the review, section 3)"},
        "K5": "RECORD: n = 1, 2 cells' L, Z per cell (active.avg - q u), L0 and mean concurrency; tp4 and tp2 w1 Z_f / u against npn",
        "identification": {
            "H_ZT": "K1, K2 and K4 hold and K3 reads the ns form: a per-GEMM transient of about 0.4 u inside SM-busy time, constant in nanoseconds; a term a_g = zeta u_g / f is then registered for rental 3, never adopted from these pages",
            "H_LD": "K1 fails: the constant is launch/drain, and L0 says how much of it any kernel pays",
            "H_IMB": "K4 fails: the constant is dispatch imbalance",
            "else": "INCONCLUSIVE, with each K printed"},
        "seen_data": ["a = elapsed - ceil(q) u at tp2 and tp4 n <= 9 is SEEN (rental-1 r3c pages hold NATIVE sm__cycles_elapsed at the 1710 lock); only the active and shape counters are new",
                      "tp8's floor counters at n = 1..6, 8, 10 are SEEN (rental 1)"],
    }
    # ---- part 4
    SETS = {"mixtral-8x7b-tp8": {"w1": {"A": (13, 14, 15), "B": (11, 15, 16), "C": (11, 12, 13)},
                                 "w2": {"W2a": (11, 12, 13), "W2b": (14, 15, 16)}},
            "mixtral-8x7b-tp4": {"w1": {"A": (14, 15, 16), "B": (11, 13, 15), "C": (10, 12, 13)},
                                 "w2": {"W2a": (11, 12, 13), "W2b": (14, 15, 16)}},
            "mixtral-8x7b-tp2": {"w1": {"D1": (10, 11, 12), "D2": (12, 13, 14)},
                                 "w2": {"W2a": (11, 12, 13), "W2b": (14, 15, 16)}}}
    sets_out, theta_noise = {}, {}
    for model, gs in SETS.items():
        for g, ss in gs.items():
            gg = geo[model][g]
            pc = {c["n"]: (c["grid"], c["cores"]) for c in cells[model] if c["gemm"] == g}
            grids = {n: v[0] for n, v in pc.items()}
            rows4 = {}
            for name, ns in ss.items():
                assert all(next(c for c in cells[model] if c["gemm"] == g and c["n"] == n)["waves"] >= 4
                           for n in ns), (model, g, ns)
                est = FE.implied_slope(pc, ns, gg["ksteps"])
                w = FE.slope_weights(grids, ns, gg["ksteps"])
                rows4[name] = {"treads": list(ns), "H_EST": r6(est), "bias": r6(est / gg["asymptote"] - 1),
                               "FLUID": gg["asymptote"], "FLUID_LOW": r6(0.964 * gg["asymptote"]),
                               "sigma_slope": r6(sigma_cell * FE.norm(w))}
            sets_out[f"{model} {g}"] = {"ksteps": gg["ksteps"], "npn": gg["npn"], "asymptote": gg["asymptote"],
                                        "u": gg["u"], "sets": rows4}
            if g == "w1" and len(ss) >= 3:
                sl = [{"treads": v["treads"], "asymptote": gg["asymptote"], "bias": v["bias"]} for v in rows4.values()]
                wt, wd = FE.theta_delta_weights(sl, grids, gg["ksteps"])
                theta_noise[f"{model} w1"] = {"sigma_theta": r6(sigma_cell * FE.norm(wt)),
                                              "sigma_delta": r6(sigma_cell * FE.norm(wd))}
            elif g == "w1":
                # tp2: delta only, theta fixed by each family
                ws = [FE.slope_weights(grids, v["treads"], gg["ksteps"]) for v in rows4.values()]
                comb = {}
                for w in ws:
                    for n, x in w.items():
                        comb[n] = comb.get(n, 0.0) + x / gg["asymptote"] / len(ws)
                theta_noise[f"{model} w1"] = {"sigma_delta": r6(sigma_cell * FE.norm(comb))}

    def npn_deficit(npn):
        return -0.036 + (0.036 - 0.007) * math.log(npn / 56) / math.log(448 / 56)
    seen_rep = {}
    pc8 = {c["n"]: (c["grid"], c["cores"]) for c in cells["mixtral-8x7b-tp8"] if c["gemm"] == "w1"}
    seen_rep["tp8 w1 (6, 8, 10) SEEN"] = r6(FE.implied_slope(pc8, (6, 8, 10), 64))
    # the co-primary per-cell test's expected separation
    sep = {}
    for model in CAPS:
        for g in ("w1", "w2"):
            u = geo[model][g]["u"]
            un = [c for c in cells[model] if c["gemm"] == g and c["waves"] >= 4
                  and c["n"] >= UNSEEN_MIN[model]]
            ceilX = [c["ceil"] * u for c in un]
            fluidX = [c["q"] * u for c in un]
            sep[f"{model} {g}"] = {"cells": [c["n"] for c in un],
                                   "rms_fluid_if_ceil_true": r6(FE.form_rms(ceilX, fluidX)),
                                   "rms_ceil_if_fluid_true": r6(FE.form_rms(fluidX, ceilX))}
    w1f = {
        "registered": f"{CM.DATE}, before any rental-2 page",
        "name": CM.NAMES["w1floor"],
        "status": "REGISTERED BEFORE ANY PAGE",
        "design": "design-r2/DESIGN.md section 4, replaced as the review directs (REVIEW.md sections 4 and 5.4): unseen tread sets, FLUID_LOW as the rival, a co-primary per-cell test, noise-derived bands, the primary capture named",
        "tool": "scripts/floor_estimator.py; scorer scripts/scoring/rental2/score_w1floor.py",
        "primary_capture": "the BASE-clock G = 64 captures (r3f-g64.json) of the tp8 floor2, tp4 floor and tp2 floor units; the 1710 lock captures are printed beside, as the 2026-09-30 registration did",
        "x_grid": "grid x S / 132, grid = live + dead CTAs from the page's own launch__grid_size, asserted equal to the registered grid (V1 already holds it)",
        "geometry": {m: geo[m] for m in CAPS},
        "sets": sets_out,
        "sets_note": "every tread in these sets is >= 10 on tp4/tp2 and >= 11 on tp8: on no published page (rental-1 r3c pages hold n = 1..9 at G = 64 on every shard, and tp8's floor n = 1..6, 8, 10). tp2 w1 has only two unseen sets, of bias about 0 and +3.8%: delta only. Every w2 set within 10..16 sits at +3.8% (ceil(512 n / 132) does not wrap there): w2 is an offset test, H_EST +3.8% against FLUID 0%",
        "seen_replication": seen_rep,
        "noise": {"sigma_cell_cycles": sigma_cell, "source": "CAL: the median over the four calibration models' G = 64 base and lock captures of the rms of (elapsed - CORES) about a free line in grid S / 132, floor-bound cells of >= 4 waves", "theta_delta": theta_noise,
                  "propagation": "every slope, theta and delta is linear in the cells' cycles (floor_estimator.slope_weights, theta_delta_weights): sigma = sigma_cell x ||weights||, exact for sets that share cells; the fit is unweighted OLS over the sets"},
        "families": {
            "H_EST": "the CORES model through F4's estimator: theta 1, delta 0",
            "FLUID": "a set-independent slope at the asymptote 344.1 + F / S: theta 0, delta 0",
            "FLUID_LOW": "the FINDINGS reading of rental 1: a set-independent slope of 0.964 x the asymptote: theta 0, delta -3.6%",
            "H_NPN": "secondary, labelled: H_EST x (1 + delta_npn), delta_npn -3.6% at npn 56 log-linear to -0.7% at 448 (already refuted on SEEN data: it predicts 327.3 on (6, 8, 10) where 339.6 was measured)",
            "npn_deficit": {m: r6(npn_deficit(geo[m]["w1"]["npn"])) for m in CAPS}},
        "rules": {
            "fit": "per w1 GEMM with three sets (tp8, tp4): r_k = slope_k / asymptote - 1 = delta + theta x bias_k, unweighted OLS over the sets (floor_estimator.theta_delta); tp2 w1: delta_H = mean(r_k - bias_k) and delta_F = mean(r_k)",
            "theta_call": "theta > 0.5 reads H_EST's side, theta < 0.5 FLUID's",
            "H_EST": "FALSIFIED if |theta - 1| > 2 sigma_theta on tp8 w1 or tp4 w1, or |delta_H| > 2 sigma_delta on two of the three w1 GEMMs",
            "FLUID": "FALSIFIED if |theta| > 2 sigma_theta on tp8 w1 or tp4 w1, or |delta_F| > 2 sigma_delta on two of the three w1 GEMMs",
            "FLUID_LOW": "FALSIFIED if |theta| > 2 sigma_theta on tp8 w1 or tp4 w1, or |delta_F + 0.036| > 2 sigma_delta on tp8 w1",
            "H_NPN": "FALSIFIED if delta_H on tp8 w1 > -2.0% (secondary)",
            "w2_offset": "per shard, the mean over W2a, W2b of slope / asymptote - 1: H_EST predicts the mean registered bias (+3.8%), FLUID 0; each family HOLDS within 2 sigma (sigma the mean's propagated noise / asymptote), else FALSIFIED",
            "co_primary_per_cell": "per GEMM, on the unseen cells of >= 4 waves (n >= 10 on tp4/tp2, n >= 11 on tp8): rms_C of cycles about ceil(q) u + a and rms_F about q u + a, a one free constant each; CEIL wins when rms_F - rms_C > sigma_cell, FLUID when rms_C - rms_F > sigma_cell, else INCONCLUSIVE. H_EST is supported when CEIL wins on tp8 w1 and tp4 w1; FLUID (and FLUID_LOW) when FLUID wins on both",
            "both_fail": "when every family is FALSIFIED the verdict is NEITHER: no family is selected and the per-set slopes are printed",
            "per_cell_record": "each >= 4-wave cell's (measured - CORES) / CORES is printed beside part 3's a"},
        "co_primary_expected_separation": sep,
        "power": {k: {"theta_bands_2sigma": {"H_EST": [r6(1 - 2 * v["sigma_theta"]), r6(1 + 2 * v["sigma_theta"])],
                                             "FLUID": [r6(-2 * v["sigma_theta"]), r6(2 * v["sigma_theta"])]},
                      "disjoint": bool(1 - 2 * v["sigma_theta"] > 2 * v["sigma_theta"])}
                  for k, v in theta_noise.items() if "sigma_theta" in v},
        "power_note": "theta's 2-sigma bands overlap on tp8 w1 (sigma_theta 0.28 at sigma_cell 2.7k cycles over 3-tread sets) and are disjoint on tp4 w1; the co-primary per-cell test (about 2.5 sigma_cell of rms separation per GEMM) carries the call where theta cannot",
        "unseen_min_tread": UNSEEN_MIN,
        "seen_data": ["tp8 w1 (6, 8, 10) and every tread <= 9 at G = 64 on all three shards are SEEN; none is in a scored set",
                      "H_NPN's -3.6% is read off rental-1 tp8 (SEEN) and 8x7B's -0.7%"],
    }
    return const, w1f


# --------------------------------------------------------------------------
# the .txt beside each JSON: the tables a person reads, rendered from the JSON
# --------------------------------------------------------------------------

def _f(v, p=3):
    return "--" if v is None else f"{v:.{p}f}"


def txt_knobs(d: dict) -> list[str]:
    L = [f"REGISTERED {d['registered']}: rental 2 part 1, B's knobs (lag law L2r) and C's BLOCK_K separator T5",
         "s at G = 1 (scripts/l2_survival.py); every knob page scored against its same-board default page", "",
         "lambda(n), SEEN (rental 1's tp2 w2 / tp4 w1 pair), +- sigma from +-0.03 on each s in quadrature;",
         "the steep rival ST is the registered 2026-10-01 lambda (CAL):"]
    for n, v in d["law_L2r"]["lambda_by_n_SEEN"].items():
        L.append(f"  n={n}  lambda {v['lam']:.3f} +- {v['sigma']:.3f} ({v['rel'] * 100:.0f}%)   steep "
                 f"{d['rivals']['lambda_steep_by_n'][n]:.4f}   s_cap at x 0.576 "
                 f"{d['law_L2r']['s_cap_knots_by_n'][n][1][1]:.3f}" + ("   (printed only)" if int(n) < 4 else ""))
    L += ["", "bands, illustrated on rental 1's measured s as s_base (SEEN; the scores use the same-board base page):"]
    for key, v in d["illustrations_on_rental1_SEEN_bases"].items():
        L.append(f"  {key}, dd {v['dd']:.3f}")
        for n in ("4", "5", "6", "7", "8", "9"):
            b = v["bands"][n]
            L.append(f"    n={n}  L2r [{b['L2r'][0]:.3f}, {b['L2r'][1]:.3f}]  NL [{b['NL'][0]:.3f}, {b['NL'][1]:.3f}]  "
                     f"ST [{b['ST'][0]:.3f}, {b['ST'][1]:.3f}]")
    L += ["", "tests:"]
    for k, v in d["tests"].items():
        L.append(f"  {k}: {v['rule']}")
    L += ["", "controls:"]
    for k, v in d["controls"].items():
        L.append(f"  {k}: {v if isinstance(v, str) else v['rule'] + (' ' + v['effect'] if 'effect' in v else '')}")
    pc = d["partition_cross_check"]
    L += ["", f"partition cross-check: {pc['rule']}; {pc['effect']}", "",
          f"T5: {d['T5']['what']}", f"  f: {d['T5']['f']}", f"  pricing: {d['T5']['pricing']}",
          f"  band: {d['T5']['band']}", f"  rule: {d['T5']['rule']}",
          f"  board check: {d['T5']['board_check']['rule']}; {d['T5']['board_check']['effect']}"]
    for k, v in d["T5"]["registered_at_default_Wc"].items():
        L.append(f"  {k:4s} ({d['T5']['labels'][k]}) f(n=9) BK32 {v['f9']['32']:.4f} BK64 {v['f9']['64']:.4f} "
                 f"BK128 {v['f9']['128']:.4f}  rho_BK {v['rho_BK']:.3f}  band [{v['band_at_default_Wc'][0]:.3f}, "
                 f"{v['band_at_default_Wc'][1]:.3f}]")
    L.append("  RECORD, tp2 G=64 s8 (W_c 264) f(n=9): " + ", ".join(f"{k} {v:.3f}" for k, v in d["T5"]["s8_record_at_Wc264"].items()))
    L += ["", "seen data:"] + [f"  {s}" for s in d["seen_data"]]
    return L


def txt_launch2(d: dict) -> list[str]:
    L = [f"REGISTERED {d['registered']}: rental 2 part 2, the launch-floor rerun (two processes per model)",
         f"tool: {d['tool']}", ""]
    for m, v in d["models"].items():
        L.append(f"  {m}: {v}")
    L += ["", "C_reg (ms, G = 4), mixtral-8x7b-tp8, from cross_model_predict on 8x7B's 2026-09-27 pages (CAL):"]
    for arm, row in d["C_reg_ms"]["mixtral-8x7b-tp8"].items():
        L.append(f"  {arm:8s} " + " ".join(f"{row[str(n)]:.4f}" for n in range(1, 10)))
    L += ["", f"F: E240 {d['F_ms']['E240']} ms, E480 {d['F_ms']['E480']} ms ({d['F_ms']['basis']})", "", "definitions:"]
    for k, v in d["definitions"].items():
        L.append(f"  {k}: {v}")
    L += ["", "P0 fails if: " + "; ".join(d["P0"]["fails_if"]) + f". {d['P0']['effect']}."]
    for p in ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"):
        v = d[p]
        L.append(f"{p}: cells {v.get('cells', '')}; {v.get('rule', '')}" + (f" [{v['label']}]" if isinstance(v.get("label"), str) else ""))
    L += ["", "predicted host-bound tp8 cells (printed estimates; the scores use each cell's H_pre):"]
    for h in ("H_0.30", "H_0.35"):
        for mode, arms in d["predicted_host_bound_tp8"][h].items():
            L.append(f"  {h} {mode}: " + "; ".join(f"{a} n {v}" for a, v in arms.items()))
    L += ["", "seen data:"] + [f"  {s}" for s in d["seen_data"]]
    return L


def txt_const(d: dict) -> list[str]:
    L = [f"REGISTERED {d['registered']}: rental 2 part 3, the missing per-GEMM constant",
         f"captures: " + "; ".join(f"{k} {v['label']} treads {v['treads']} {v['clocks']}" for k, v in d["captures"].items() if isinstance(v, dict)),
         f"primary: {d['captures']['primary']}", "", "definitions:"]
    for k, v in d["definitions"].items():
        L.append(f"  {k}: {v}")
    L += ["", "CAL, Z by intercept (sm__cycles_active.avg on q, the slope free), floor-bound cells of >= 2 waves:"]
    for r in d["cal"]["series"]:
        if r["cal"]:
            L.append(f"  {r['model']:16s} {r['session'][:10]} {r['clock']:4s} {r['gemm']} cells {r['cells']}  Z/u {r['Zu']:+.3f}  "
                     f"slope/u {r['slope_over_u']:.4f}  clock {r['mhz']:.0f} MHz  active rms {r['active_rms']:.0f} cycles")
    L.append(f"  sigma_active (median of the CAL series' rms): {d['cal']['sigma_active_cycles']:.0f} cycles")
    L += ["", f"K1: {d['K1']['rule']} ({d['K1']['basis']})",
          f"K2: {d['K2']['cells']}; band {d['K2']['band']}, centre {d['K2']['centre']:.3f}; {d['K2']['rule']} ({d['K2']['basis']})",
          f"K3: {d['K3']['what']}", f"  forms: ns {d['K3']['forms']['ns']}; cycles {d['K3']['forms']['cycles']}",
          f"  rule: {d['K3']['rule']}"]
    for g, v in d["K3"]["noise"].items():
        L.append(f"  {g}: cells at 1005 {v['cells_1005']}, at 1710 {v['cells_1710']}; sigma_Z {v['sigma_Z_1005']:.0f} / "
                 f"{v['sigma_Z_1710']:.0f} cycles at Z {v['Z_assumed']:.0f}: sigma_r / r {v['sigma_ratio_rel']:.3f}")
    L += [f"  {d['K3']['cal_gamma_record']}", f"K4: {d['K4']['rule']}; {d['K4']['cta_clause']}", f"K5: {d['K5']}", "",
          "identification:"] + [f"  {k}: {v}" for k, v in d["identification"].items()]
    L += ["", "registered floor-bound cells (sigma 1), per capture model:"]
    for m, cs in d["cells_registered"].items():
        for g in ("w1", "w2"):
            L.append(f"  {m} {g}: n " + ", ".join(str(c["n"]) for c in cs if c["gemm"] == g and c["floor_bound"]))
    L += ["", "seen data:"] + [f"  {s}" for s in d["seen_data"]]
    return L


def txt_w1floor(d: dict) -> list[str]:
    L = [f"REGISTERED {d['registered']}: rental 2 part 4, tp8's w1 floor on unseen tread sets",
         f"primary capture: {d['primary_capture']}", f"x grid: {d['x_grid']}", "",
         "per set: the slope over grid x S / 132 (F4's estimator); H_EST = the CORES model's cells through it (CAL)"]
    for key, s in d["sets"].items():
        L.append(f"  {key} (S {s['ksteps']}, npn {s['npn']}, asymptote {s['asymptote']})")
        for name, v in s["sets"].items():
            L.append(f"    {name} {tuple(v['treads'])}: H_EST {v['H_EST']:.1f} (bias {v['bias'] * 100:+.2f}%)  FLUID "
                     f"{v['FLUID']:.1f}  FLUID_LOW {v['FLUID_LOW']:.1f}  sigma {v['sigma_slope']:.1f}")
    L += [f"  {d['sets_note']}", f"  SEEN replication: {d['seen_replication']}", "",
          f"noise: sigma_cell {d['noise']['sigma_cell_cycles']:.0f} cycles ({d['noise']['source']})"]
    for k, v in d["noise"]["theta_delta"].items():
        L.append(f"  {k}: " + ", ".join(f"{a} {b:.4f}" for a, b in v.items()))
    L += [f"  {d['noise']['propagation']}", "", "families:"] + [
        f"  {k}: {v}" for k, v in d["families"].items() if k != "npn_deficit"]
    L.append("  H_NPN deficits: " + ", ".join(f"{m} {v * 100:+.2f}%" for m, v in d["families"]["npn_deficit"].items()))
    L += ["", "rules:"] + [f"  {k}: {v}" for k, v in d["rules"].items()]
    L += ["", "power:"] + [f"  {k}: theta 2-sigma bands H_EST {v['theta_bands_2sigma']['H_EST']}, FLUID "
                           f"{v['theta_bands_2sigma']['FLUID']}, disjoint {v['disjoint']}" for k, v in d["power"].items()]
    L += [f"  {d['power_note']}", "", "co-primary test, expected rms of the wrong form about the right one (cycles):"]
    for k, v in d["co_primary_expected_separation"].items():
        L.append(f"  {k}: cells {v['cells']}, {v['rms_fluid_if_ceil_true']:.0f}")
    L += ["", "seen data:"] + [f"  {s}" for s in d["seen_data"]]
    return L


TXT = {"knobs": txt_knobs, "launch2": txt_launch2, "const": txt_const, "w1floor": txt_w1floor}


def build() -> dict:
    const, w1f = const_and_w1floor()
    return {"knobs": knobs(), "launch2": launch2(), "const": const, "w1floor": w1f}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=REG)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    docs = build()
    bad = 0
    for part, doc in docs.items():
        for path, text in ((a.out_dir / f"{CM.NAMES[part]}.json", json.dumps(doc, indent=1, default=str) + "\n"),
                           (a.out_dir / f"{CM.NAMES[part]}.txt", "\n".join(TXT[part](doc)) + "\n")):
            if a.check:
                same = path.exists() and path.read_text() == text
                print(f"{path.name}: {'same' if same else 'DIFFERS'}")
                bad += not same
            else:
                path.write_text(text)
                print(f"wrote {path}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
