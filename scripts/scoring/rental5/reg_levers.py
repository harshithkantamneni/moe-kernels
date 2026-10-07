"""Rental 5's lever registrations: v2 (no unit), secondk and bk128 (tp2 CUT by the owner, 2026-10-07)
(docs/registered/2026-10-07-rental5-{v2,secondk,bk128}-gh200), built for
scripts/scoring/rental5/register.py, which renders and --checks them.

Every number is computed here from committed files (the timing model under v2, rental 4's
register.py fits, r4common's exposure and the published pages) or is a typed-in input with
its provenance and label (CAL, SEEN, SEEN-fitted, BLIND, BLIND-CALMODEL). Design:
scratchpad design-r5/DESIGN.md parts 1 and 4, corrected by design-r5-review/REVIEW.md
sections (a) and (f); the review wins where they conflict.

LEAKAGE, said once for the four: every published page is SEEN; every constant typed in below
was fitted or chosen after seeing published pages (labelled per entry). No rental-5 page exists
when this runs.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r5common as C5  # noqa: E402

C4 = C5.C4
r6 = C5.r6
REPO = C5.REPO
R4REG = C5._load("rental4_register", HERE.parent / "rental4" / "register.py")

#: the expected v2 rms table (%, model M then v2): design-r5/OFFSET.md sections 2 and 3, own
#: counted bytes, SHARED + PRIVATE under rental 3's host rule. SEEN. score_v2seen reproduces
#: it to 0.01 point or exits 1.
EXPECTED_V2 = {
    "rental3 E2 (counted)": (2.52, 0.97),
    "rental3 E1 (predicted)": (3.66, 1.79),
    "qwen2-57b-a14b": (1.06, 1.13),
    "olmoe-1b-7b": (1.40, 0.88),
    "jetmoe-8b": (0.84, 0.86),
    "qwen1.5-moe-a2.7b": (1.87, 1.28),
    "phi-3.5-moe": (0.57, 0.58),
    "mixtral-8x22b": (1.03, 0.99),
    "mixtral-8x7b (in-sample)": (0.26, 0.26),
}
#: rental 4's per-GEMM d, printed there (SEEN): d_w2 from A1, d_w1 from A2
DG = {"d_w1": 0.95, "d_w2": 1.10, "label": "SEEN: rental 4's printed per-GEMM d (A1 d_w2, A2 d_w1); a printed rival column, never a model"}
D2 = dict(R4REG.D2)
CAL_COUNTERS = "CAL-counters (4 models: 8x7B, 8x22B, Qwen2-57B, OLMoE)"


_FITS: dict = {}


def source_fit(variant: str):
    """The 8x7B 2026-09-27 source fit (CAL) under dead model `variant`: (x, k_w, params)."""
    if variant not in _FITS:
        import cores_heldout_predict as CP
        import r3_timing_model as TM
        old = TM.set_model("mixtral-8x7b")
        try:
            with TM.dead_model(variant):
                b = TM.build(TM.build_parser().parse_args([*map(str, CP.source_pages()), "--counters", str(CP.C27)]))
        finally:
            TM.set_model(old)
        _FITS[variant] = (b["main"].x, b["main"].k_w, b["main"].params)
    return _FITS[variant]


def _v2_fit() -> dict:
    return {v: {k: r6(x) for k, x in source_fit(v)[2].items()} for v in ("m", "v2")}


def v2() -> dict:
    import r3_timing_model as TM
    D = R4REG.fit_D()
    fits = _v2_fit()
    return {
        "registered": f"{C5.DATE}, before any rental-5 page",
        "name": C5.NAMES["v2"],
        "status": "REGISTERED MODEL CHANGE (no unit, no GPU): model v2; its re-predictions of earlier tests are SEEN DIAGNOSTIC, no verdict",
        "design": "design-r5 DESIGN.md part 1 with REVIEW.md section (a) (CAL-counters label, IN-SAMPLE labels, OLMoE's BLIND-CALMODEL basis)",
        "tool": "scripts/r3_timing_model.py --dead-model v2 (dead_model('v2')); scorer scripts/scoring/rental5/score_v2seen.py",
        "model": {"v2": "M with D: d = V2_DEAD_NS per dead CTA replaces DEAD_CTA_NS 1.333; kappa = V2_KAPPA replaces k_w in the dead window, hidden = kappa occ_g S'_g c; T0, c, bw, s_small, s_block refit on the CAL 8x7B 2026-09-27 pages",
                  "constants": {"V2_DEAD_NS": TM.V2_DEAD_NS, "V2_KAPPA": TM.V2_KAPPA},
                  "version": TM.MODEL_VERSION["v2"],
                  "d_kappa_fit": {**D, "set": CAL_COUNTERS,
                                  "label": "CAL-counters: fitted on the in-kernel SHARED - NATIVE gap of the four CAL models' counter pages (rental 4 register.py fit_D, from offset/work/deadfit.json rows); 8x22B, Qwen2-57B and OLMoE are also held-out TIME-test models, their counter pages SEEN"},
                  "refit_8x7b": {"M": fits["m"], "v2": fits["v2"],
                                 "label": "CAL: the five 2026-09-27 mixtral_8x7b timed pages (G 2, 3, 4, 8, 32) and 2026-09-27-r3-counters/lock1710, the cross-model source set; no held-out page enters"},
                  "no_new_constant": "v2 adds kappa only; d replaces DEAD_CTA_NS",
                  "kappa_reading": "the tp2 lever (9 against 32 copies on mixtral-8x7b-tp2) was CUT by the owner on 2026-10-07; kappa is read from the stamps only (docs/registered/2026-10-07-rental5-stamps2-gh200, D2 hiding)"},
        "recorded_not_adopted": {"Dg": DG, "D2": D2,
                                 "rule": "each enters only as a printed rival column (Dg, D2) in score_v2seen and the rental-5 dead rivals"},
        "flagged": {"M1_model_change": "a --dead-model {m,v2} switch in r3_timing_model; the default stays m, so every earlier scorer's output is byte-identical",
                    "code_pins": "r3_timing_model.py's sha changes; the rental-4 registrations' code_pins record the old sha (85ef38c), those registrations were scored and are not re-run"},
        "re_predictions": {"tests": ["cross_model_score targets: 8x22B, Qwen2-57B, OLMoE, Qwen1.5, Phi-3.5, JetMoE (and Granite-3.0-3B, printed with no expected value)",
                                     "rental 3 E2 (counted bytes) and E1 (registered q_pred)", "rental 4 A1 and A2 (measured Delta beside M's and v2's)",
                                     "jetmoe-floor and mixtral-floor: floor slopes in cycles per CTA k-step, no dead term: unaffected"],
                           "each_twice": "as registered (model M) and under v2, the 8x7B source refit under each",
                           "set": "SHARED + PRIVATE under rental 3's host rule (T_pred + F240 >= 0.40 ms, F240 0.067551 ms)",
                           "printed": "rms, worst, bias (mean e), per-arm mean us, and r5common.secondary (calibration in logs, descriptive below 6 clusters; gamma)",
                           "labels": {"mixtral-8x7b": "CAL (in-sample)",
                                      "mixtral-8x22b, qwen2-57b-a14b, olmoe-1b-7b": "IN-SAMPLE (d, kappa): their counter pages are in the CAL-counters set, so v2's dead term is in-sample on them; NOT held-out",
                                      "others": "held-out, SEEN"},
                           "verdict": "NONE: a SEEN diagnostic"},
        "expected_rms_pct": {k: list(v) for k, v in EXPECTED_V2.items()},
        "expected_rule": "score_v2seen must reproduce every expected value (M and v2) to 0.01 point, or it exits 1",
        "labels": {"olmoe_skew": "OLMoE's skew cells are BLIND-CALMODEL only because OLMoE's counter pages are in the CAL-counters set; no OLMoE time page enters any v2 fit"},
        "seen_data": ["every page read is SEEN", "d, kappa are CAL-counters (fitted on four models' counter pages, three of them held-out time models)",
                      "the expected table is SEEN (OFFSET.md, computed on published pages)"],
    }


# --------------------------------------------------------------------------
# secondk: per-CTA fixed against main loop at a second K
# --------------------------------------------------------------------------

#: OLMoE rental-4 counter c per k-step (SEEN, rental 4 SCORES.md part B'), and CAL F
C64 = {"w1": 346.2, "w2": 344.0}
C32 = {"w1": 245.2, "w2": 234.3}
K_OLMOE = {"w1": 2048, "w2": 1024}
SECONDK_MODEL = "qwen2-57b-a14b-tp8"


def secondk_pred() -> dict:
    from moe.spec import MODEL_CONFIGS
    cfg = MODEL_CONFIGS[SECONDK_MODEL]
    K = {"w1": cfg.hidden_size, "w2": cfg.intermediate_size}
    out = {}
    for g in ("w1", "w2"):
        F = C4.F_CTA[g]
        S64, S32 = K[g] // 64, K[g] // 32
        if K[g] % 32 or K[g] % 64:
            raise ValueError(f"{g}: K {K[g]} is not whole k-steps at BK 32 and 64")
        u64 = S64 * C64[g] + F
        it = (S32 * C32[g] + F) / u64
        dF = (C32[g] - C64[g] / 2) * (K_OLMOE[g] // 32)
        cta = (S32 * C64[g] / 2 + F + dF) / u64
        out[g] = {"K": K[g], "S64": S64, "S32": S32, "H_ITER": r6(it), "H_CTA": r6(cta), "dF_cycles": r6(dF),
                  "gap_pct": r6(100 * abs(it / cta - 1))}
    return out


def secondk() -> dict:
    return {
        "registered": f"{C5.DATE}, before any rental-5 page",
        "name": C5.NAMES["secondk"],
        "status": "REGISTERED BEFORE ANY PAGE",
        "design": "design-r5 DESIGN.md part 4b with REVIEW.md section (f) (H_ITER and H_CTA stay separated even if F at S = 5 is off by 50%); second K placed ABOVE BK 128 in the plan",
        "tool": "scorer scripts/scoring/rental5/score_secondk.py; r4common.page_c",
        "pages": {"path": f"<date>-<card>-{SECONDK_MODEL}-<label>-r3-counters/lock1710/r3c-g64.json", "labels": {"k64s4": [64, 4], "k32s4": [32, 4]},
                  "design_check": "each page's design (block_k, num_stages) must equal its label's; else NOT SCORED",
                  "k_note": "w2 K = 320 at BK 32 is 10 whole k-steps (at BK 64, 5), so neither page is refused"},
        "statistic": {"c_page": "per (page, GEMM) c by r4common.page_c over NATIVE and SHARED cells n = 4..9 at DRAM <= 0.5 x 4022 GB/s, F held at CAL (w1 520, w2 979 clk); fewer than 3 cells: NOT SCORED",
                      "r_meas": "u(k32) / u(k64), u = S c + F",
                      "e": "r_meas / r_pred - 1 per hypothesis"},
        "inputs": {"c64": C64, "c32": C32, "K_olmoe": K_OLMOE, "F_cal": dict(C4.F_CTA),
                   "label": "c64 / c32 SEEN (OLMoE rental-4 B' counter pages, SCORES.md); F CAL (r3_timing_model CTA_FIXED_KSTEPS)"},
        "hypotheses": {"H_ITER": "the BK 32 excess is per k-step: u(k32) = S32 c32 + F",
                       "H_CTA": "the same excess is per CTA: u(k32) = S32 c64 / 2 + F + dF, dF = (c32 - c64/2) x OLMoE's BK-32 steps"},
        "predicted_ratio": secondk_pred(),
        "rules": {"per_gemm": "the nearer hypothesis within 3% (|e| <= 3%) is that GEMM's reading; both beyond 3%: NEITHER",
                  "verdict": "SELECTED <H> when both GEMMs read the same H; else UNDECIDED (the readings printed)",
                  "noise": "counter noise about 0.3% against gaps of 14% (w1) and 29% (w2)"},
        "gates": "the counter page's V-gates by rental 2's two views; V6, V7 and V10 do not gate (cycles are read per GEMM)",
        "missing": "a page missing or out of the view: NOT SCORED; the drop-group k2 keeps the pair together",
        "labels": {"k64s4": "SEEN (rental 3's qwen2-57b-a14b-tp8 G = 64 floor pages ran k64s4)", "k32s4": "BLIND"},
        "secondary": "r5common.secondary on (r_pred of the selected-or-nearest H, r_meas): NOT APPLICABLE below 3 values",
        "seen_data": ["c64, c32 are SEEN (OLMoE rental 4)", "F is CAL", "the k64s4 configuration is SEEN on this model; k32s4 is BLIND"],
    }


# --------------------------------------------------------------------------
# bk128: BK 128 timed, a blind confirmation of the occupancy-3 floor
# --------------------------------------------------------------------------

BK_MODEL = "olmoe-1b-7b"
BK_NS = (4, 5, 6, 7, 8, 9)
#: rental 4's SEEN OLMoE G = 64 counter c per k-step (SCORES.md part B'): BK 128 s4 at
#: occupancy 3 / 3, BK 64 s4 at 5 / 4
C128_HC = {"w1": 686.9, "w2": 684.1}
C64_BASE = {"w1": 346.2, "w2": 344.0}


def _native_ms(n: int, fitx, k_w: float) -> float:
    """OLMoE NATIVE G = 64 at tread n under v2 and the 8x7B v2 fit, sigma = 1 (floor-bound)."""
    import r3_timing_model as TM
    ctx = TM.Context(card="nvidia_gh200_480gb", device="", sms=C5.SMS, sms_source="registered", occupancy={"w1": 5, "w2": 4},
                     occupancy_source="SEEN cubins (k64s4)", clock_mhz=C5.LOCK_MHZ, locked=True, bandwidth_gbps=3725.1,
                     byte_label="GROUPMODEL")
    path = TM.SMALL_BATCH if TM.numel_of(n) < 1024 and TM.E <= 64 else TM.BLOCK_SCAN
    cell = TM.Cell(arm="native", G=64, n=n, ms=0.0, mhz=C5.LOCK_MHZ, path=path, declared=TM.E, runs=(),
                   source="GROUPMODEL", reads=None, sigma={"w1": 1.0, "w2": 1.0}, fit=False)
    with TM.dead_model("v2"):
        return TM.call_ms(fitx, cell, ctx, k_w)


def bk128_pred() -> dict:
    import r3_timing_model as TM
    from moe.spec import MODEL_CONFIGS
    cfg = MODEL_CONFIGS[BK_MODEL]
    K = {"w1": cfg.hidden_size, "w2": cfg.intermediate_size}
    x, kw, _ = source_fit("v2")
    old = TM.set_model(BK_MODEL)
    try:
        rivals = {"H_C": C128_HC, "SYNC": {g: C64_BASE[g] for g in K}, "PS": {g: 2 * 344.1 for g in K}}
        out = {}
        for n in BK_NS:
            T64 = _native_ms(n, x, kw)
            row = {"T64_ms": r6(T64)}
            for name, c128 in rivals.items():
                dt = 0.0
                for g in ("w1", "w2"):
                    live = TM.live_rows(n) * TM.GEOMETRY[g].npn
                    q = math.ceil(live / C5.SMS)
                    f64 = q * ((K[g] // 64) * C64_BASE[g] + C4.F_CTA[g])
                    f128 = q * ((K[g] // 128) * c128[g] + C4.F_CTA[g])
                    dt += (f128 - f64) / (C5.LOCK_MHZ * 1e3)
                row[name] = r6((T64 + dt) / T64)
            out[str(n)] = row
    finally:
        TM.set_model(old)
    return out


def bk128() -> dict:
    return {
        "registered": f"{C5.DATE}, before any rental-5 page",
        "name": C5.NAMES["bk128"],
        "status": "REGISTERED BEFORE ANY PAGE: a BLIND confirmation of the occupancy-3 floor, not a lever",
        "design": "design-r5 DESIGN.md part 4a with REVIEW.md section (f): SYNC (q 0.55-0.65) is already excluded by the SEEN counters (c128 686.9 is about 2 x 346), so this confirms the floor's time blind; num-stages=4 explicit on both pages; placed BELOW second K",
        "tool": "scorer scripts/scoring/rental5/score_bk128.py",
        "pages": {"path": f"gaps-<card>-{BK_MODEL}-<label>/private_weight_reference/*/report.json, the G = 64 page, NATIVE only, treads 1..9",
                  "labels": {"bk64": [64, 4], "bk128": [128, 4]},
                  "design_check": "each page's pinned BLOCK_SIZE_K and num_stages must equal its label's; else NOT SCORED"},
        "statistic": {"q_n": "T(bk128, n) / T(bk64, n), NATIVE ms_p50, same board, n = 4..9 (floor-bound)", "e": "q_meas / q_pred - 1"},
        "floor_formula": "q_pred = 1 + sum_g ceil(N_live,g / 132) [(S128 c128 + F_g) - (S64 c64 + F_g)] / (1710 MHz) / T64_pred, T64_pred the v2 timing model at sigma 1 (OLMoE NATIVE G 64, the 8x7B v2 fit), c64 the SEEN k64s4 counter c (346.2 / 344.0)",
        "rivals": {"H_C": {"c128": C128_HC, "label": "SEEN rental-4 counters at BK 128 s4, occupancy 3 / 3"},
                   "SYNC": {"c128": dict(C64_BASE), "label": "c per k-step held at BK 64: the floor halves"},
                   "PS": {"c128": 2 * 344.1, "label": "c128 = 2 x 344.1 (CAL c), printed"}},
        "predicted_q": bk128_pred(),
        "rules": {"H_C": "HOLDS if rms e <= 1% over the scored n; else FAILS",
                  "SYNC": f"EXCLUDED if |e_SYNC| > 3 sqrt2 sigma_page ({r6(3 * math.sqrt(2) * C5.SIGMA_PAGE)}) on at least 4 cells",
                  "PS": "printed only (0.2 to 0.6% from H_C)",
                  "need": "at least 4 of n = 4..9 on both pages, else NOT SCORED"},
        "gates": "R3's VALIDITY gates by rental 3's two views (ALL and CLEAN); V5 does not gate",
        "missing": "a page missing: NOT SCORED; the drop-group bk keeps the pair together",
        "labels": {"time": "BLIND-CALMODEL (OLMoE is a CAL model; no OLMoE G = 64 timed page at BK 128)", "counters": "SEEN"},
        "secondary": "r5common.secondary on (q_pred H_C, q_meas), one cluster: calibration NOT RESOLVED",
        "seen_data": ["c128, c64 SEEN (rental 4 counters)", "the v2 fit is CAL (8x7B)"],
    }


BUILDERS = {"v2": v2, "secondk": secondk, "bk128": bk128}


def _txt_v2(d):
    return ["expected rms (M -> v2, %): " + "; ".join(f"{k} {a:.2f} -> {b:.2f}" for k, (a, b) in d["expected_rms_pct"].items())]


def _txt_secondk(d):
    return ["predicted u(k32)/u(k64): " + "; ".join(f"{g} H_ITER {v['H_ITER']:.4f} H_CTA {v['H_CTA']:.4f} (gap {v['gap_pct']:.1f}%)"
                                                     for g, v in d["predicted_ratio"].items())]


def _txt_bk128(d):
    return ["predicted q = T128/T64: " + "; ".join(f"n{n} H_C {v['H_C']:.4f} SYNC {v['SYNC']:.4f} PS {v['PS']:.4f}"
                                                   for n, v in d["predicted_q"].items())]


TEXT = {"v2": _txt_v2, "secondk": _txt_secondk, "bk128": _txt_bk128}
