"""Rental 5's skew and c15 registrations (docs/registered/2026-10-07-rental5-skew-gh200,
-c15-gh200): every prediction computed here, from committed files, before any page.

Design: scratchpad design-r5/DESIGN.md part 2, corrected by design-r5-review/REVIEW.md
(b, c, d, g, h and owner item 1, whose G3 text is registered verbatim) and by the owner's
decisions of 2026-10-07 (1 G3 as a TOST with a cluster-t interval and Cochran's Q, every
per-cell difference kept, the model's shuffle effect registered per cell, the permutation
cells included; 3 SYNTHETIC skew: Zipf / Dirichlet shapes fitted to the traces' statistics,
drawn from registered seeds, published openly, the predictions recomputed from the draws).
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r5common as C5  # noqa: E402
import skewmodel as SM  # noqa: E402

REPO = C5.REPO
HIST = REPO / "docs" / "registered" / "2026-10-07-rental5-skew-hist"
SKEW_MODELS = ("mixtral-8x7b", "olmoe-1b-7b", "qwen1.5-moe-a2.7b")
TIMED_PAGES = ("A", "B")
BYTE_MODELS = ("mixtral-8x7b", "olmoe-1b-7b")
SKEWED = ("PT", "PW", "DW")
PERMS = ("PW-hotfirst", "PW-rand1")
#: the treads past every timed CAL tread (R3's DEFAULT_TREADS 6): E1-UNI-EXT (review b3)
EXT_TREAD = 6
LABELS = {"mixtral-8x7b": "BLIND-CALMODEL (8x7B is the CAL timing model; no skewed GH200 page exists)",
          "olmoe-1b-7b": "BLIND-CALMODEL only because OLMoE is in the CAL-counters set (8x7B, 8x22B, Qwen2-57B, OLMoE) that d and kappa were fitted on (review a3)",
          "qwen1.5-moe-a2.7b": "BLIND (a SEEN held-out model, not CAL)"}
#: build-r5-review F3: the balanced cells at n <= 6 (G 8, BLOCK_M 32, NATIVE and SHARED at 9
#: copies) are on published pages, so they are not blind
BALANCED_SEEN = {"mixtral-8x7b": "CAL (SEEN): the 2026-09-27 GH200 8x7B timed pages (G 8, n 1..6) carry this balanced cell",
                 "olmoe-1b-7b": "SEEN: the 2026-09-29 and 2026-10-07 (rental 4) OLMoE G 8 timed pages carry this balanced cell",
                 "qwen1.5-moe-a2.7b": "SEEN: the 2026-09-29 Qwen1.5 G 8 timed pages carry this balanced cell"}
#: the rival lever thresholds (design-r5 2.5, review d)
LEVER_U_PW = 0.045       # 3 x the 1.5% prior on S
LEVER_LT = 0.0075        # 3 x 0.25% (one cell against one, sqrt2 x 0.18%)
r6 = C5.r6


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha_counts(counts) -> str:
    return hashlib.sha256(json.dumps(None if counts is None else [int(v) for v in counts],
                                     separators=(",", ":")).encode()).hexdigest()


def page_file(model: str, page: str) -> Path:
    return HIST / f"{model}-{page}.json"


def load(model: str, page: str) -> dict:
    return json.loads(page_file(model, page).read_text())


_ENGINE = None


def engine() -> SM.Engine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = SM.Engine()
    return _ENGINE


def _arms(c: dict) -> list:
    return list(c.get("arms") or ("native", "shared"))


def timed_cells(e: SM.Engine) -> list[dict]:
    rows = []
    for model in SKEW_MODELS:
        for page in TIMED_PAGES:
            doc = load(model, page)
            for c in doc["cells"]:
                for arm in _arms(c):
                    p = e.price(model, arm, c["n"], c["counts"])
                    rows.append({"model": model, "page": page, "n": c["n"], "label": c["label"],
                                 "arm": arm, "counts_sha256": sha_counts(c["counts"]),
                                 "declared": p.get("declared"), "path": p.get("path"),
                                 "live_rows": p.get("live_rows"), "dead_ctas": p.get("dead_ctas"),
                                 "q_pred": None if p["q"] is None else {g: r6(v) for g, v in p["q"].items()},
                                 "ms": {k: r6(p[k]) for k in ("S", "LT", "PW", "M1")},
                                 "ext_tread": c["n"] > EXT_TREAD,
                                 "label_blind": (BALANCED_SEEN[model] if c["label"] == "balanced" and c["n"] <= EXT_TREAD
                                                 else LABELS[model])})
    by = {(r["model"], r["page"], r["n"], r["label"], r["arm"]): r for r in rows}
    for r in rows:
        u = by.get((r["model"], r["page"], r["n"], "uniform", r["arm"]))
        if r["label"] in SKEWED and u and all(r["ms"][k] and u["ms"][k] for k in ("S", "LT", "PW", "M1")):
            r["ratio"] = {k: r6(r["ms"][k] / u["ms"][k]) for k in ("S", "LT", "PW", "M1")}
            r["ratio"]["U"] = 1.0
            rs = r["ratio"]["S"]
            r["lever"] = {"U": abs(1.0 - rs) > LEVER_U_PW, "PW": abs(r["ratio"]["PW"] - rs) > LEVER_U_PW,
                          "LT": abs(r["ratio"]["LT"] - rs) > LEVER_LT}
        if r["label"] in PERMS:
            pw = by.get((r["model"], r["page"], r["n"], "PW", r["arm"]))
            if pw and pw["ms"]["S"] and r["ms"]["S"]:
                r["perm_dr_S"] = r6(r["ms"]["S"] / pw["ms"]["S"] - 1)
        if r["label"] == "uniform":
            b = by.get((r["model"], r["page"], r["n"], "balanced", r["arm"]))
            if b and b["ms"]["S"] and r["ms"]["S"]:
                r["delta_M3"] = r6(r["ms"]["S"] / b["ms"]["S"] - 1)
    return rows


def byte_cells(e: SM.Engine) -> list[dict]:
    rows = []
    for model in BYTE_MODELS:
        doc = load(model, "C")
        for c in doc["cells"]:
            for arm in _arms(c):
                q = e.wsc_q(model, arm, c["n"], None if c["counts"] is None else tuple(c["counts"]))
                rows.append({"model": model, "page": "C", "n": c["n"], "label": c["label"], "arm": arm,
                             "counts_sha256": sha_counts(c["counts"]),
                             "q_pred": None if q is None else {g: r6(v) for g, v in q.items()},
                             "label_blind": LABELS[model]})
    return rows


def histogram_files() -> dict:
    out = {}
    for p in sorted(HIST.glob("*.json")):
        d = json.loads(p.read_text())
        ent = {"path": str(p.relative_to(REPO)), "sha256": sha256_file(p)}
        if "cells" in d:
            ent["shuffle_seed"] = d["shuffle_seed"]
            ent["cells"] = [{"label": c["label"], "n": c["n"], "arms": _arms(c),
                             "counts_sha256": sha_counts(c["counts"])} for c in d["cells"]]
        out[p.name] = ent
    return out


def lever_lists(rows: list) -> dict:
    out = {}
    for k in ("U", "PW", "LT"):
        out[k] = [f"{r['model']}/{r['page']}/n{r['n']}/{r['label']}/{r['arm']}" for r in rows
                  if r.get("lever", {}).get(k)]
    return out


G3_TEXT = ("G3 (per board). d_i = T(uniform-shuffled)/T(balanced) - 1 - delta_M3,i, on every valid "
           "(shape, page, n, arm) cell, NATIVE and SHARED pooled. Equivalence holds if the 90% cluster-t "
           "CI of mean(d), clustered on page (df = pages - 1), lies within +-0.36%. Heterogeneity: Q = sum "
           "(d_i - mean)^2 / (2 x 0.0018^2) against chi2(N-1). If p < 0.01, apply the same TOST per "
           "n-stratum, and only the failing strata's histogram cells are voided. Otherwise, a failure voids "
           "every histogram cell on the board. Every d_i, Q, its p, and the per-shape and per-arm means are "
           "printed. The same rule is applied to the permutation cells against S's predicted dr.")


def skew() -> dict:
    e = engine()
    rows = timed_cells(e)
    brows = byte_cells(e)
    ratio_rows = [r for r in rows if "ratio" in r]
    levers = lever_lists(ratio_rows)

    def rng(model, key):
        v = [100 * (r["ratio"][key] - 1) for r in ratio_rows if r["model"] == model]
        return [r6(min(v)), r6(max(v))] if v else None
    summary = {m: {"S_minus_1_pct": rng(m, "S"), "LT_minus_1_pct": rng(m, "LT"),
                   "PW_minus_1_pct": rng(m, "PW")} for m in SKEW_MODELS}
    synth = json.loads((HIST / "synth_fit.json").read_text())
    return {
        "registered": f"{C5.DATE}, before any rental-5 page",
        "name": C5.NAMES["skew"],
        "status": "REGISTERED BEFORE ANY PAGE",
        "design": "design-r5 DESIGN.md part 2 corrected by design-r5-review REVIEW.md (b, c, d, g, h; owner item 1's G3 text verbatim) and the owner's decisions of 2026-10-07 (1 G3 TOST + cluster-t + Cochran Q, permutation cells; 3 SYNTHETIC histograms fitted to the traces' statistics, published openly)",
        "tool": "scorer scripts/scoring/rental5/score_skew.py; predictions scripts/scoring/rental5/skewmodel.py (r3_timing_model MODEL M2, wave_split_bytes MODEL M3) and reg_skew.py; histograms scripts/skew_synth.py",
        "model": {"S": "model v2 (docs/registered/2026-10-07-rental5-v2-gh200) with MODEL M2 (r3_timing_model.schedule with counts: live rows sum_e ceil(c_e / 32), owner = the expert whose tile range holds pid_m, A share per actual rows, grid_rows and dead_ctas from numel, B_OTHER proportional to numel) and MODEL M3 (wave_split_bytes.walk with counts: the per-expert owner in the walk and in each group's working set, stack_distance on each group's distinct owners, equal to an exact LRU walk (tests); A re-reads weighted by their tile's rows; no closed form)",
                  "U": "skew ignored: priced at the uniform histogram of the same tokens; r_U = 1",
                  "LT": "S without the last wave's quantisation (wave_q = 1): the close rival (review d)",
                  "PW": "S plus each expert's CTAs quantised to their own waves (the grouped-GEMM view), applied to every cell including uniform and balanced, so r_PW is its own ratio",
                  "M1": "S under model M's dead term (DEAD_CTA_NS 1.333, k_w 0.5), printed only",
                  "NATIVE_bytes": "NATIVE's bytes are SHARED's walk (the byte model has no dead-CTA term), as in every cross-model registration",
                  "source": e.source_doc()},
        "histograms": {"synthetic": "SYNTHETIC (owner decision 3): Zipf / Dirichlet shapes fitted by scripts/skew_synth.py to summary statistics (cv, max/mean, Gini) of three public routing logs, drawn from registered seeds; no trace-derived count is in the repository or on the VM; the pages are published openly",
                       "files": histogram_files(),
                       "shapes": [{k: f[k] for k in ("model", "n", "B", "shape", "kind", "param", "loss", "target", "fit_residual", "matches", "trace_statistic")} for f in synth["shapes"]],
                       "coverage": "the shapes span the typical (median-CV) to the most skewed layer of each scheme, at the per-layer MEDIAN over batches (p50); they do not cover the p90 batches of any layer or the least-skewed layers (build-r5-review section 2). Mixtral DW at n 8 and 32 use the registered least-loss fallback (cv -11.6%, Gini +15% off the target)",
                       "match_rule": f"a statistic is MATCHED when the fitted shape's median over {synth['fit_seeds']} seeds is within {synth['match_tol']:.0%} of the trace statistic",
                       "draw_rule": "per shape the first of 64 registered seeds whose draw has all three statistics within 10% of the target, else the draw of least loss (scripts/skew_synth.py --draw)",
                       "shuffle": "every non-balanced cell: realize_counts(counts, tokens, k), then ids[torch.randperm(tokens, Generator().manual_seed(shuffle_seed))]; 'balanced' is R3's balanced_ids unshuffled"},
        "cells": {"common": "R3's tile 32x64x64 w8 s4 at G = 8; NATIVE and SHARED at 9 copies (this is not vLLM's tile, so nothing here is the working bar)",
                  "pages": {"A": "PT, PW (NATIVE, SHARED), PW-hotfirst and PW-rand1 (NATIVE only), uniform and balanced (NATIVE, SHARED), at the model's three n",
                            "B": "DW, uniform, balanced (NATIVE, SHARED) at the three n",
                            "C": "the byte leg (counter page, Mixtral and OLMoE): PT, PW, uniform, balanced at n <= 9 (NATIVE, SHARED) and OLMoE n 16 NATIVE only"},
                  "treads": {"mixtral-8x7b": [2, 8, 32], "olmoe-1b-7b": [2, 4, 16], "qwen1.5-moe-a2.7b": [1, 2, 8]},
                  "n_stratum": "G3's n-stratum is the tread's rank within its shape (first, second, third n: Mixtral 2 / 8 / 32, OLMoE 2 / 4 / 16, Qwen1.5 1 / 2 / 8), so every stratum keeps all six pages as clusters",
                  "labels": LABELS},
        "predictions": rows,
        "byte_predictions": brows,
        "summary": summary,
        "levers": {"rule": f"U and PW: |r_rival - r_S| > {LEVER_U_PW:.1%} (3 x the 1.5% prior); LT: |r_LT - r_S| > {LEVER_LT:.2%} (3 x 0.25%, one cell against one)",
                   "cells": levers, "counts": {k: len(v) for k, v in levers.items()}},
        "gates": {"source": "docs/registered/2026-10-06-rental4-nativegates-gh200 (owner decision 1 of 2026-10-06) with G3 replaced below",
                  "G1_lock_thermal": "V7: the 1710 lock in force and the thermal gate, as every timed page",
                  "G2_host_bound": "R3's host-bound guard: a cell whose host enqueue exceeds its GPU time is excluded (the page's histogram_gates lists them)",
                  "G4_worst_cell_clock": "every scored cell's measured SM clock is the 1710 lock: a cell read below 1705 MHz (a third of the 15 MHz grid step under it) is excluded (OWNER-CONFIRMED 2026-10-07)",
                  "G5_provenance": "the page's histogram file sha256 equals the registered one, every row's counts_sha256 equals its registered cell's, bincount_ok on every row, the shuffle seed recorded; else the page is NOT SCORED",
                  "G3": G3_TEXT,
                  "G3_what_it_tests": "OWNER DECISION 2026-10-07: G3 is kept as a test of TOKEN ORDER against the model's predicted zero effect. The uniform-shuffled and balanced cells have identical per-expert counts and identical per-expert token sets; they differ only in which tokens sit in which rows (the row shuffle), which moves the A-row gather locality. Every registered model (M2, M3 and the content-keyed term) predicts delta_M3 = 0 for that, so G3 asks whether token order alone moves the time by more than the +-0.36% margin",
                  "G3_delta_M3": "registered per cell in predictions[].delta_M3 (S(uniform)/S(balanced) - 1). It is 0.0 on every cell: a token-row permutation keeps every expert's token set (the realize_counts classes of k experts with identical token sets survive it, permuted), and the registered byte view keys A by M-tile, so neither M2 nor M3 nor the content-keyed term moves. The review's 'the shuffle destroys those classes' does not hold for a row shuffle; G3 therefore tests the token ORDER (A-row gather locality) the models do not price",
                  "G3_power": "Monte Carlo of the registered g3_rule (build-r5-review area5/g3_mc.txt), 6 pages x 6 cells at sigma_page 0.18%: with no effect a board is voided 0.4% of the time; with a true effect at the 0.36% margin it wrongly passes 5.2% of the time; Q's size 1.1%; a +0.6% effect at one n is caught by Q 93% of the time. If the true cell noise is 0.25%, a board with no effect is voided 38% of the time (Q fires on the noise itself)",
                  "G3_margin_fixed": "the margin stays fixed at +-0.36% (2 x 0.18%) whatever sigma the pages show; it is never re-chosen on a page",
                  "G3_no_rows": "a board with no valid G3 row (no uniform-balanced pair on any page) has no control: every histogram cell's test (SKEW-RATIO, E1-SKEW, the permutation control, B-SKEW, E2-SKEW) is NOT SCORED (build-r5-review F6); E1-UNI and E1-UNI-EXT, on balanced cells, are still scored",
                  "G3_margin": C5.G3_MARGIN, "G3_q_alpha": C5.Q_ALPHA, "G3_shared": "SHARED is pooled with NATIVE in G3 (the review's text); the per-arm means are printed",
                  "views": "R3's VALIDITY gates by rental 3's two views (ALL and CLEAN, rental 2's addendum rule 3); a page whose V1 failed is unusable in both",
                  "perm_control": "the G3 rule on the permutation cells: d_i = [T(perm)/T(PW) - 1]_meas - perm_dr_S, NATIVE, clustered on page; a failure is printed as S failing the label-permutation control (it voids nothing)"},
        "tests": {
            "SKEW-RATIO": {"statistic": "r = T(h) / T(uniform-hist), same page, n and arm; e = r_pred / r_meas - 1 on every valid (shape, page, n, h in PT PW DW, arm); a ratio needs both cells valid",
                           "S": "HOLDS if rms(e) <= 1.5% and max|e| <= 4%; else FAILS",
                           "rivals": "U, PW and LT are each EXCLUDED if their rms on their own lever cells is at least 2 x S's rms there and at least 3 sqrt2 sigma_page (0.76%); UNDECIDED if fewer than 6 of a rival's lever cells survive the gates; M1 printed",
                           "rms_max": 0.015, "max_abs": 0.04, "rival_factor": 2.0, "rival_floor": round(3 * math.sqrt(2) * C5.SIGMA_PAGE, 6), "min_lever_cells": 6},
            "E1-SKEW": {"statistic": "e = T_pred / T_meas - 1, predicted bytes (M3), on every valid histogram cell (PT PW DW, the permutations, uniform)",
                        "rule": "HOLDS if rms <= 2.5%, worst <= 6% and |mean e| <= 1.5%", "rms_max": 0.025, "max_abs": 0.06, "mean_max": 0.015,
                        "prior": "2 to 3%: rental 3's E1 under v2 reads 1.79% (SEEN)"},
            "E1-UNI-EXT": {"statistic": "e on the balanced cells at treads past every timed CAL tread (Mixtral 32, OLMoE 16, Qwen1.5 8; R3's DEFAULT_TREADS 6), reported apart from E1-SKEW (review b3)",
                           "rule": "the E1-SKEW bands, printed as HOLDS or FALSIFIED; the balanced cells at n <= 6 are printed as E1-UNI (in range)"},
            "B-SKEW": {"statistic": "e_B = q_pred / q_counted - 1 per (cell, GEMM) on the byte-leg counter pages (q = (dram_bytes_read - n x operand) / W_g)",
                       "rule": "HOLDS if every PT / PW cell's |e_B| <= 5% (the WSC registrations' per-cell band) and their rms <= 3%; the uniform and balanced cells of the same page are printed as the control",
                       "gates": "the byte leg takes the timed pages' gates too (build-r5-review F5): G5 row by row (every counter cell's counts_sha256 equals its registered cell's and its bincount equals the counts, else the page is NOT SCORED), G3's voiding (a voided stratum's byte cells are dropped, and no G3 row means NOT SCORED), and the 1705 MHz cut on each cell's SM clock = sm__cycles_elapsed.avg / gpu_time_ns over its (cell, GEMM) of at least 0.5 ms (shorter GEMMs read low by a fixed offset, rental 3), else the page's median over such GEMMs; a page with no such GEMM is NOT SCORED",
                       "max_abs": 0.05, "rms_max": 0.03, "owner": "thresholds OWNER-CONFIRMED 2026-10-07 as registered"},
            "E2-SKEW": {"statistic": "e = T(counted bytes) / T_meas - 1: S's time with sigma from the byte-leg page's counted bytes of the same (model, n, label, arm), computed by skewmodel.Engine.price_counted at the registered fit and code",
                        "rule": "HOLDS if rms <= 2%, worst <= 5% and |mean| <= 1.5%", "rms_max": 0.02, "max_abs": 0.05, "mean_max": 0.015,
                        "owner": "thresholds OWNER-CONFIRMED 2026-10-07 as registered",
                        "code": "scored only when scripts/r3_timing_model.py, scripts/wave_split_bytes.py and scripts/scoring/rental5/skewmodel.py have the registration's sha256 (code_pins)"},
            "DECOMPOSITION": "per byte-leg cell, ln(T_pred / T_meas) = ln(T_pred / T_counted) + ln(T_counted / T_meas): the byte model's part (B as time) and E2's, printed (review c)",
            "reported": "per arm, per shape and pooled; ALL and CLEAN; the calibration line (cluster = shape), gamma (strata (shape, arm, n) across PT PW DW uniform; mostly skewed-against-uniform pairs resolve), top-1 regret NOT APPLICABLE"},
        "noise": {"sigma_page": C5.SIGMA_PAGE, "sigma_page_alt": C5.SIGMA_PAGE_ALT,
                  "rule": "every rule at 0.18% and also printed at 0.10% (rental 4's NATIVE c9 / c15 replicate)"},
        "missing": "a ratio needs both its cells valid; a missing page's cells are NOT SCORED; the drop-groups skm, sko and skq keep each shape's A, B (and C) pages together",
        "rival_note": "PW sits far above S on every cell (a strawman, as the design said); LT is the close rival (review d): its lever cells are listed",
        "wsc_citation": "the per-expert walk is in scripts/wave_split_bytes.py (walk, stack_distance, tile_content_ids, DESIGN_KEYS at c213664 lines 615 (owner at 638), 590, 788, 1093), not r3_timing_model's 590-640 CORES block (review b)",
        "leakage": {"no_skewed_gh200_page": "no skewed GH200 page exists; every R3 page so far is balanced_ids",
                    "padding_seen": "docs/FINDINGS.md:2729 'Padding is either zero or free' (SEEN, H200, the August sweeps): above batch 256 vLLM's autotuner sizes BLOCK_SIZE_M to rows per expert (0% padding), and below it the wasted arithmetic hides inside a 20 us weight read. S predicts the opposite here, +2 to +30% from padding. The regime boundary: R3 pins BLOCK_M 32 (no autotuner, so a partly filled tile is real), and these cells are floor-bound (the CTA's shared-memory floor, not the weight read, sets its time), where a partly filled M-tile costs a whole CTA lifetime",
                    "earlier_skew_timings": "the August H200 / A100 sweeps timed zipf, hot and dirichlet (FINDINGS C5 'seven routing regimes'): SEEN, another harness, unlocked, fitted by no current constant",
                    "histograms": "the synthetic shapes are fitted to SEEN routing statistics (no GH200 page); the counts are drawn, never timed, fitted on nothing",
                    "wsc": "WSC's card constants are CAL on uniform 8x7B pages only; a skew page is a target only and never joins a CAL card",
                    "d_kappa": "d and kappa are fitted on the CAL-counters set (8x7B, 8x22B, Qwen2-57B, OLMoE counter pages): OLMoE's cells are BLIND-CALMODEL only because of that set"},
        "seen_data": ["the timing fit is 8x7B's CAL 2026-09-27 pages under v2; d and kappa CAL-counters", "the byte model's parameters are CAL (8x7B counter pages)",
                      "sigma_page 0.18% and 0.10% are SEEN", "the routing statistics the shapes are fitted to are SEEN (routing logs, no timing)",
                      "no skewed GH200 page exists: every cell is BLIND or BLIND-CALMODEL (labels)"],
    }


C15_SIMS = 40000


def c15_sigma(T_us: float, sigma: float = C5.SIGMA_PAGE, sims: int = C15_SIMS) -> float:
    """sd of median_h,n Delta(h, n) with one NATIVE uniform cell per (page, n): c15's N15(n) and
    c9's N9 from page A (PT, PW, uniform) or B (DW), every cell's noise sigma x T."""
    import numpy as np
    rng = np.random.default_rng(C5.SEED)
    s = sigma * T_us
    meds = np.empty(sims)
    D = np.empty((sims, 8))
    i = 0
    for _n in (2, 4):
        n15, n9a, n9b = (rng.normal(0, s, sims) for _ in range(3))
        for h in ("PT", "PW", "DW", "uniform"):
            n9 = n9b if h == "DW" else n9a
            D[:, i] = (rng.normal(0, s, sims) - n15) - (rng.normal(0, s, sims) - n9)
            i += 1
    meds = np.median(D, axis=1)
    return float(meds.std())


def c15() -> dict:
    e = engine()
    doc = load("olmoe-1b-7b", "c15")
    model = "olmoe-1b-7b"
    rows = []
    for c in doc["cells"]:
        if "shared" not in _arms(c):
            continue
        p9 = e.price(model, "shared", c["n"], c["counts"], 9)
        p15 = e.price(model, "shared", c["n"], c["counts"], 15)
        rows.append({"label": c["label"], "n": c["n"], "counts_sha256": sha_counts(c["counts"]),
                     "S9_ms": r6(p9["S"]), "S15_ms": r6(p15["S"]),
                     "dead_increment_ctas": {g: p15["dead_ctas"][g] - p9["dead_ctas"][g] for g in ("w1", "w2")},
                     "dead9_total": sum(p9["dead_ctas"].values()), "dead15_total": sum(p15["dead_ctas"].values()),
                     "live_rows": p9["live_rows"],
                     "Delta_us": {"D": r6((p15["S"] - p9["S"]) * 1e3), "M": r6((p15["M1"] - p9["M1"]) * 1e3), "FIXED": 0.0}})
    T = st.median(r["S15_ms"] for r in rows)
    sig_indep = 1.25 * 2 * C5.SIGMA_PAGE * T / math.sqrt(len(rows)) * 1e3
    sig = c15_sigma(T * 1e3)
    dD = st.median(r["Delta_us"]["D"] for r in rows)
    dM = st.median(r["Delta_us"]["M"] for r in rows)
    return {
        "registered": f"{C5.DATE}, before any rental-5 page",
        "name": C5.NAMES["c15"],
        "status": "REGISTERED BEFORE ANY PAGE",
        "design": "design-r5 DESIGN.md 2.6 (the SHARED c15 lever) with design-r5-review (c15 kept; the c15 skew cells are BLIND-CALMODEL)",
        "tool": "scorer scripts/scoring/rental5/score_c15.py",
        "pages": {"c15": "olmoe-1b-7b timed label sk15: SHARED at 15 copies on PT PW DW uniform, NATIVE uniform, n 2 and 4 (docs/registered/2026-10-07-rental5-skew-hist/olmoe-1b-7b-c15.json)",
                  "c9": "the skew pages ska (PT, PW, uniform) and skb (DW) of olmoe-1b-7b at 9 copies",
                  "declaration_check": "each page's recorded experts_declared for SHARED must be 64 x its label's copies (15 or 9), else NOT SCORED"},
        "why": "at fixed n the SHARED dead term and the live (NATIVE-path) term are collinear on the c9 skew cells (design-r5 work/collin_skew.txt: r -0.999); with the c15 cells r is about -0.08",
        "statistic": {"Delta": "Delta(h, n) = [S - N](c15) - [S - N](c9), N the NATIVE uniform cell of each page, us; the alignment growth A_n (SHARED - NATIVE align_probe, c15 minus c9, rental 4's) subtracted when both pages carry an align_probe, else raw Delta, labelled",
                      "median": "the median over the 8 cells (PT PW DW uniform x n 2, 4)",
                      "two_regressor": "[S - N] = a + d N_dead + b live over the c9 and c15 cells, printed with its VIF"},
        "predictions": rows,
        "predicted_median_us": {"D": r6(dD), "M": r6(dM), "FIXED": 0.0},
        "noise": {"sigma_noise_us": r6(sig),
                  "basis": f"the sd of the median of the 8 Deltas by simulation (c15_sigma: {C15_SIMS} draws, seed {C5.SEED}), every cell at 0.18% of T = the median predicted S(c15) {T:.4f} ms, with ONE NATIVE uniform cell per (page, n) shared by the four histograms (build-r5-review F4: the Deltas are correlated)",
                  "independent_formula_us": r6(sig_indep), "independent_formula": "1.25 x 2 x 0.18% x T / sqrt 8 (the design's, which treats the 8 Deltas as independent): printed only", "review_check": "build-r5-review's own simulation read 1.292 us and D against M 6.2 sigma; this registration's seeded simulation reads the value above"},
        "z": "(Delta_meas - Delta_pred) / sigma_noise",
        "rules": {"D": "FAILS if |z_D| > 3; else HOLDS", "M": "EXCLUDED if |z_M| > 3",
                  "FIXED": "EXCLUDED if Delta > 3 sigma_noise", "SKEW-DEAD": "D predicts the dead increment is the same at every h (it depends on numel and the declaration only); SKEW-DEAD is shown when |median_h Delta(h in PT PW DW) - Delta(uniform)| > 3 sigma_noise",
                  "separation": f"D against M {abs(dD - dM) / sig:.1f} sigma, D against FIXED {abs(dD) / sig:.1f} sigma",
                  "printed": "every Delta(h, n), raw and alignment-corrected, both noise values (0.18% and 0.10%)"},
        "labels": "BLIND-CALMODEL: the c15 skew cells have no page; A2's uniform c15 page (rental 4) is SEEN, so D's level is SEEN-confirmed there and only the skew invariance is blind",
        "secondary": "calibration and gamma: NOT APPLICABLE (one statistic over 8 cells; fewer than 3 predicted values are distinct)",
        "missing": "the c15 page or either c9 page missing: NOT SCORED (drop-group c15 holds sk15; the c9 pages are ska / skb of sko, never dropped)",
        "seen_data": ["d, kappa CAL-counters; the fit CAL", "A2's uniform c15 page SEEN", "sigma_page SEEN"],
    }


BUILDERS = {"skew": skew, "c15": c15}


def _txt_skew(d: dict) -> list[str]:
    out = ["predicted skew ratio r_S - 1 (%), range per shape:"]
    for m, v in d["summary"].items():
        out.append(f"  {m}: S {v['S_minus_1_pct']}  LT {v['LT_minus_1_pct']}  PW {v['PW_minus_1_pct']}")
    out.append(f"lever cells: {d['levers']['counts']}")
    return out


def _txt_c15(d: dict) -> list[str]:
    return [f"predicted median Delta (us): {d['predicted_median_us']}  sigma_noise {d['noise']['sigma_noise_us']} us"]


TEXT = {"skew": _txt_skew, "c15": _txt_c15}
