#!/usr/bin/env python
"""Writes rental 3's six registered JSON files (docs/registered/2026-10-05-rental3-*).

    python scripts/scoring/rental3/register.py [--out-dir docs/registered] [--check]

Every number is computed here from committed files: the calibration models'
published pages (CAL), every other published page where a number is SEEN
(labelled; every published page is SEEN, rental 2 and the parent Qwen2-57B
pages included), and the registered rental-2 files. Nothing is typed in but the
registered rules and their thresholds. `--check` recomputes and compares with
the committed JSON and text (exit 1 on any difference) instead of writing; the
`code_pins` block (sha256 of the code a scorer imports) is excluded from the
comparison, because it records the registration commit's files and the scorer
checks it at score time. Takes about two minutes on a laptop CPU (the 8x7B
source fit is built three times: plain, under --gemm-const, and inside
cross_model_predict).

The design is scratchpad design-r3/DESIGN.md; every fix of design-r3-review/
REVIEW.md is in, and where the review corrects the design the review wins.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r3common as C3  # noqa: E402

REPO = C3.REPO
PUB = REPO / "results" / "published"
REG = REPO / "docs" / "registered"
R2S = PUB / "2026-10-02-nvidia_gh200_480gb-rental2-session" / "results"
PARENT = PUB / "2026-09-28-nvidia_gh200_480gb-qwen2-57b-session" / "results"
S25 = PUB / "2026-09-25-nvidia_gh200_480gb-session"
S25_RUNS = ("d9f1f37c", "df37ea07", "01c08abd", "1b285de2", "6ff34777")
CAL_MODELS = {"mixtral-8x7b", "mixtral-8x22b", "qwen2-57b-a14b", "olmoe-1b-7b"}
TARGET = C3.TARGET
r6 = C3.r6

#: the captures of the plan (scripts/plans/rental3-2026-10.plan), per model: label -> clocks
TREADS = {
    "qwen2-57b-a14b-tp8": [2, 3, 4, 5, 6, 8, 10, 12, 13, 14, 16],
    "granite-3.0-1b-a400m": [3, 4, 5, 6, 8, 10, 12, 14, 16],
    "mixtral-8x7b-tp4": [2, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16],
    "mixtral-8x7b-tp8": [6, 8, 10, 11, 12, 13, 14, 15, 16],
    "mixtral-8x7b-tp2": [2, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16],
}
CAPTURES = {
    "qwen2-57b-a14b-tp8": {"floor": ["base", "lock1710"], "floor1005": ["lock1005"]},
    "granite-3.0-1b-a400m": {"floor": ["base", "lock1710"], "floor1005": ["lock1005"]},
    "mixtral-8x7b-tp4": {"floor": ["base", "lock1710"], "floorrep": ["lock1710"],
                         "floor1005": ["lock1005"], "floor1410": ["lock1410"]},
    "mixtral-8x7b-tp8": {"floor1005": ["lock1005"]},
    "mixtral-8x7b-tp2": {"floor1005": ["lock1005"]},
}
#: the host rule (DESIGN 2.3): F240 the 240 MiB flush over the 3.7261 TB/s ruler, and
#: 0.40 ms the top of the measured eager host time on four models (0.33 to 0.36) plus a margin
F240_MS = 251.7e6 / 3.7261e12 * 1e3
HOST_MS = 0.40
TIMED_CORE = {8: range(1, 7), 32: range(1, 7), 3: range(1, 9)}
TIMED_DEEP = {4: range(1, 10), 2: range(1, 10)}
BYTE_G = (3, 8, 32)
IDENT = {"max_corr": 0.95, "min_cells": 5}
SE_THETA_MAX = 0.15


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# shared computations
# --------------------------------------------------------------------------

class Ctx:
    """The 8x7B 2026-09-27 source fit (CAL), the target's predictions, the CAL
    floor series, each computed once."""

    def __init__(self):
        import cores_heldout_predict as CP
        import cross_model_predict as X
        import r3_timing_model as TM
        self.CP, self.X, self.TM = CP, X, TM
        self.src = TM.build(TM.build_parser().parse_args(
            [*map(str, CP.source_pages()), "--counters", str(CP.C27)]))
        self.fit, self.ctx = self.src["main"], self.src["ctx"]
        self.pred = X.predict(CP.source_pages(), CP.C27, TARGET, False, gemm_const=True)
        self.design = X.target_design(TARGET, 32)
        self._cells = {}

    def cells(self, model: str, treads) -> list[dict]:
        """The CAL CORES model's cells at these treads (G = 64 NATIVE), with q, ceil, u,
        the floor share at sigma 1 and floor_bound (share >= 0.97, waves >= 2)."""
        key = (model, tuple(treads))
        if key in self._cells:
            return self._cells[key]
        TM = self.TM
        pr = self.CP.predict(model, list(treads), [64], self.fit.params, self.fit.k_w)
        old = TM.set_model(model)
        try:
            out = []
            for c in pr["cells"]:
                g = c["gemm"]
                u = C3.u_cycles(TM.GEOMETRY[g].ksteps, g)
                q = c["live_ctas"] / C3.SMS
                share = math.ceil(q) * u / c["cores"]
                out.append(dict(n=c["n"], gemm=g, q=r6(q), ceil=math.ceil(q), frac=r6(C3.frac_of(q)),
                                u=r6(u), ksteps=TM.GEOMETRY[g].ksteps, npn=TM.GEOMETRY[g].npn,
                                waves=r6(c["waves"]), live=c["live_ctas"], dead=c["dead_ctas"],
                                grid=c["live_ctas"] + c["dead_ctas"], cores=r6(c["cores"]),
                                floor_share_sigma1=r6(share),
                                floor_bound=bool(share >= 0.97 and c["waves"] >= 2)))
        finally:
            TM.set_model(old)
        self._cells[key] = out
        return out


def capture_key(path: Path) -> str:
    """'base', 'lock1710', 'lock1005' ...: the capture's own clock, from its file stem
    (the review's fix of the design's pooling of tp4's 1710 and 1005 cells)."""
    stem = path.stem
    return "base" if stem == "r3f-g64" else stem.split("-")[-1]


def published_series(cx: Ctx) -> list[dict]:
    """Per published G = 64 floor capture (one series per CAPTURE FILE: base, each lock
    its own) and GEMM: Z by intercept over the floor-bound cells of >= 2 waves (>= 3
    cells), with the intercept's sigma at 750 cycles, the measured clock, CAL or SEEN."""
    import floor_estimator as FE
    TM = cx.TM
    rows = []
    for f in sorted(PUB.rglob("r3f-g64*.json")):
        if "unlocked" in f.stem:
            continue
        d = json.loads(f.read_text())
        model = d["plan"]["model"]
        treads = sorted({c["n"] for c in d["cells"]})
        cells = {(c["n"], c["gemm"]): c for c in cx.cells(model, treads)}
        sess = f.parent.parent.parent.name
        old = TM.set_model(model)
        try:
            for g in ("w1", "w2"):
                use = []
                for c in d["cells"]:
                    rc = cells[(c["n"], g)]
                    if rc["floor_bound"]:
                        m = c["per_gemm"][g]
                        use.append((rc["q"], m["sm__cycles_active.avg"], m.get("sm_clock_mhz"), c["n"]))
                if len(use) < 3:
                    continue
                q = [x[0] for x in use]
                Z, slope = FE.intercept(q, [x[1] for x in use])
                u = cells[(use[0][3], g)]["u"]
                rows.append(dict(model=model, session=sess, capture=capture_key(f), gemm=g,
                                 cells=len(use), n=[x[3] for x in use], u=u,
                                 S=TM.GEOMETRY[g].ksteps, npn=TM.GEOMETRY[g].npn,
                                 Z=r6(Z), Zu=r6(Z / u), sigZ750=r6(FE.intercept_sigma(q, 750.0)),
                                 mhz=r6(st.median([x[2] for x in use if x[2]])),
                                 cal=model in CAL_MODELS))
        finally:
            TM.set_model(old)
    return rows


def z_sigma(r: dict) -> float:
    """The CAL fit's per-series sigma: the intercept's error at sigma_active =
    max(250, 0.02 u), floored at 300 cycles (the capture-to-capture scatter)."""
    return max(300.0, r["sigZ750"] * max(250.0, 0.02 * r["u"]) / 750.0)


FORMS = {
    "PROP": lambda u, f: [u],
    "AFF": lambda u, f: [1.0, u],
    "MIX": lambda u, f: [f / 1710.0, u],
    "MIX3": lambda u, f: [f / 1710.0, 1.0, u],
}


def z_fits(series: list[dict]) -> dict:
    """The four forms by weighted least squares on the CAL base and 1710-lock
    series (weight 1 / sigma_Z^2), each series at its measured clock."""
    import numpy as np
    cal = [r for r in series if r["cal"] and r["capture"] in ("base", "lock1710")]
    fits = {}
    for name, X in FORMS.items():
        A = np.array([X(r["u"], r["mhz"]) for r in cal])
        y = np.array([r["Z"] for r in cal])
        w = np.array([1 / z_sigma(r) for r in cal])
        coef, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
        cov = np.linalg.inv((A * w[:, None]).T @ (A * w[:, None]))
        chi2 = float((((y - A @ coef) * w) ** 2).sum()) / max(1, len(cal) - len(coef))
        fits[name] = {"coef": [r6(v) for v in coef], "cov": [[float(x) for x in row] for row in cov],
                      "chi2_dof": r6(chi2), "series": len(cal)}
    return fits


def z_pred(fits: dict, form: str, u: float, f: float) -> float:
    return float(sum(c * x for c, x in zip(fits[form]["coef"], FORMS[form](u, f), strict=True)))


# --------------------------------------------------------------------------
# part E: end-to-end blind time on qwen2-57b-a14b-tp8
# --------------------------------------------------------------------------

def e2e(cx: Ctx) -> dict:
    import floor_estimator as FE
    TM = cx.TM
    d = cx.pred
    cells = {(c["arm"], c["G"], c["n"]): c for c in d["cells"]}

    def scored(Gs):
        out = []
        for G, ns in Gs.items():
            for arm in ("shared", "private"):
                for n in ns:
                    c = cells[(arm, G, n)]
                    out.append({"arm": arm, "G": G, "n": n, "T_M": r6(c["ms"]),
                                "T_MZ": r6(c.get("ms_gemm_const")),
                                "host_rule": bool(c["ms"] + F240_MS >= HOST_MS)})
        return out
    core, deep = scored(TIMED_CORE), scored(TIMED_DEEP)
    pop = [c for c in core if c["host_rule"]]
    pop_deep = [c for c in deep if c["host_rule"]]
    # E3: predicted q, every byte G and n, both GEMMs, both arms
    q = {f"{arm} G={G} n={n}": {g: r6(cells[(arm, G, n)]["q"][g]) for g in ("w1", "w2")}
         for arm in ("shared", "private") for G in BYTE_G for n in range(1, 10)}
    status = {f"{arm} G={G} n={n}": cells[(arm, G, n)]["status"]
              for arm in ("shared", "private") for G in BYTE_G for n in range(1, 10)}
    # E4: w2's floor cells at the 1005 capture
    tc = cx.cells(TARGET, TREADS[TARGET])
    w2 = [c for c in tc if c["gemm"] == "w2" and c["floor_bound"]]
    w1 = [c for c in tc if c["gemm"] == "w1" and c["floor_bound"]]
    u2, S2 = w2[0]["u"], w2[0]["ksteps"]
    b_f0 = S2 * C3.C_KSTEP / u2
    # E6: SHARED minus NATIVE in-kernel time, the dead-CTA term
    fit, ctx = cx.fit, cx.ctx
    old = TM.set_model(TARGET)
    try:
        dead = {}
        for g in ("w1", "w2"):
            sh = TM.dead_ms(cx.design.declared["shared"], 2, g, fit.params["c"], fit.k_w, ctx.occupancy[g])
            na = TM.dead_ms(cx.design.declared["native"], 2, g, fit.params["c"], fit.k_w, ctx.occupancy[g])
            per_n = {n: (TM.dead_ms(cx.design.declared["shared"], n, g, fit.params["c"], fit.k_w,
                                    ctx.occupancy[g])
                         - TM.dead_ms(cx.design.declared["native"], n, g, fit.params["c"], fit.k_w,
                                      ctx.occupancy[g])) for n in range(2, 10)}
            assert max(per_n.values()) - min(per_n.values()) < 1e-9
            dead[g] = {"dead_ctas_shared": TM.dead_ctas(cx.design.declared["shared"], 2, g),
                       "dead_ctas_native": TM.dead_ctas(cx.design.declared["native"], 2, g),
                       "hidden_us": r6(fit.k_w * ctx.occupancy[g] * TM.floor_ksteps(g) * fit.params["c"] * 1e-3),
                       "shared_us": r6(sh * 1e3), "native_us": r6(na * 1e3),
                       "pred_us": r6((sh - na) * 1e3)}
    finally:
        TM.set_model(old)
    parent = parent_dead_counted()
    # M+Z: the 8x7B refit with the term on
    pz = d["timing_params_gemm_const"]
    p0 = d["timing_params"]
    m_vs_mz = [r6(c["T_MZ"] / c["T_M"] - 1) for c in pop]
    seen = [
        "every published page is SEEN, rental 2's and the parent Qwen2-57B's (2026-09-28) included; the target has no cell on any of them",
        "the timing fit is 8x7B's 2026-09-27 pages alone (CAL); the byte model's registered view is fitted on the same board's 8x7B counter pages (CAL)",
        "the terms added after 2026-09-28 (DEAD_CTA_NS, LATER_MISS's capacity form, CORES) were chosen on SEEN pages, two of them on this family (the review, section 1)",
    ]
    parent_inputs = {
        "routing and declaration": "E 64, top 8, 9 copies, 576 slots (vLLM's 1024 bound passed), the alignment and sort over 576 slots at the same id counts, moe_sum at H 3584: the parent's exactly",
        "w2 grid": "N = H = 3584, npn 56, so 558 dead M-rows x 56 = 31,248 dead CTAs a SHARED or PRIVATE call: the parent's w2 count exactly (its counted SHARED - NATIVE w2 time is SEEN, E6's FIXED rival)",
        "w1 per-CTA unit": "K 3584, S 56, u = 19,790 cycles: the parent's w1 (the full model's w1 read 353.3 to 353.7 cycles per k-step at this S, FINDINGS 2026-09-28); only its grid width (npn 10 against 80) is new",
        "per-CTA fixed cost": "c 344.1, F_w1 520, F_w2 979 (CTA_FIXED_KSTEPS) were measured on four CAL models' counter pages, the parent's among them",
        "per-GEMM constant (M+Z only)": "the AFF coefficients are fitted on CAL floor captures, the parent's base and 1710 among them",
        "dead-CTA term": "DEAD_CTA_NS is measured on 8x7B's 2026-09-27 counter pages; it was ADOPTED to repair the parent's 2.71% miss (2.71 to 1.42% rms): selected on the parent",
        "LATER_MISS": "the capacity form was written for the parent's w2 SHARED (wave_split_bytes, 2026-09-29)",
        "what is not the parent's": "w2's K 320 (S 5, F = 36% of u = 2,700 cycles, the shortest CTA ever timed), w1's npn 10, and their combination's exposure; the dead term's hiding window shrinks from about 4.3 occ us to 0.8 occ us, so the term is extrapolated there (E6 tests it)",
    }
    return {
        "registered": f"{C3.DATE}, before any rental-3 page",
        "name": C3.NAMES["e2e"],
        "status": "REGISTERED BEFORE ANY PAGE: no rental-3 page exists when this is committed",
        "design": "scratchpad design-r3/DESIGN.md section 2 with design-r3-review/REVIEW.md sections 1, 5 and 6 (E); the review wins where they differ",
        "tool": "scripts/cross_model_predict.py --gemm-const at this commit (the numbers below); scorer scripts/scoring/rental3/score_e2e.py",
        "target": TARGET,
        "scope": ("a blind test of an UNSEEN SHAPE INSIDE THE 64-EXPERT, TOP-8 FAMILY on which every "
                  "post-2026-09-28 term was chosen: the 64-expert analogue of the 8x22B test. It is NOT a "
                  "test of transfer across expert counts (FINDINGS' rule: such a claim needs a model none of "
                  "the terms has seen), and E1's label says so"),
        "parent_inputs": parent_inputs,
        "source": {"model": d["source_model"], "board": d["board"], "clock_mhz": d["clock_mhz"],
                   "pages": d["timing_pages"], "timing_params": {k: r6(v) for k, v in p0.items()},
                   "p_knee": d["p_knee"], "k_w": d["k_w"], "byte_view": d["byte_view"],
                   "label": "CAL: 8x7B's 2026-09-27 fit held, nothing fitted on the target"},
        "design_of_target": d["design"],
        "shapes": d["shapes"],
        "models": {
            "M": "the model at this commit (primary): F, the dead-CTA term, LATER_MISS, CORES; the JSON's ms",
            "M0": "the 0f77622 model of the 09-29 registrations: differs from M only through CORES, inactive at every scored cell (2.9 or more waves): predicted differences under 0.3%; M vs M0 does not separate, stated not tested",
            "M+Z": "M plus the per-GEMM constant's cycle form (r3_timing_model --gemm-const) with T0 refit on 8x7B (timing_params_MZ); a near-uniform shift of about -9.6 us a call",
        },
        "timing_params_MZ": {k: r6(v) for k, v in pz.items()},
        "gemm_const": d["gemm_const"],
        "host_rule": {"rule": "a cell is scored when T_M + F240 >= 0.40 ms (T_M the JSON's prediction, never the measurement)",
                      "F240_ms": r6(F240_MS), "host_ms": HOST_MS,
                      "basis": "0.40 ms is the top of the measured eager host time on four models (0.33 to 0.36 ms, SEEN) plus a margin; F240 = 251.7 MB over the 3.7261 TB/s ruler"},
        "population": {"core": pop, "deep": pop_deep, "core_cells": len(pop), "deep_cells": len(pop_deep),
                       "excluded_by_host_rule": [f"{c['arm']} G={c['G']} n={c['n']}" for c in core + deep if not c["host_rule"]],
                       "rule": "the registered verdict population is the core cells (timed step: G = 3 n <= 8, G = 8 and 32 n <= 6) that pass the host rule, fixed here; the deep pages' cells (G = 4, 2 to n = 9) get their own printed verdict by the same rule and never change the registered one; a population with under 75% of its cells present (24 of 31) is NOT SCORED"},
        "E1": {"what": "time from PREDICTED bytes: (T_meas - T_M) / T_M per scored cell", "label": "BLIND (scope above: unseen shape, seen family)",
               "rule": "HOLDS when rms <= 2% and no cell beyond 5%; FALSIFIED otherwise", "rms_max": 0.02, "max_abs": 0.05,
               "unresolved": "beside the registered verdict, the label UNRESOLVED when |rms - 2%| or |max - 5%| is under the resolution sqrt(sigma_page^2 + sigma_board^2) (the replicate registration); the verdict stands"},
        "E2": {"what": "time from COUNTED bytes: scripts/cross_model_score.py score() at this commit on the target's timed and counter pages (the byte G's 3, 8, 32)",
               "cells": "the E1 population restricted to G with a byte page (3, 8, 32): every core cell", "rule": "the same as E1", "label": "BLIND"},
        "E3": {"what": "PRIVATE weight-set reads q (dram_counter_route.r3_q) against the JSON's q",
               "cells": "every G in (3, 8, 32), n = 1..9 (bytes are not host-bound), w1 and w2: 54 cells",
               "rule": "HOLDS when the rms of q_meas / q_pred - 1 is <= 3% and at most 10% of cells lie beyond 5%; FALSIFIED otherwise",
               "rms_max": 0.03, "beyond": 0.05, "beyond_share_max": 0.10, "label": "BLIND"},
        "E3s": {"what": "SHARED q (an expert fits in L2: q 1.00 to 1.04 predicted)", "rule": "E3's rule, SECONDARY (Qwen2 w2 SHARED missed by +97%, SEEN)", "label": "BLIND"},
        "q_pred": q, "q_status": status,
        "E4": {"what": "the floor's per-CTA unit u = S c + F on w2 (S 5, F 979: 36% of u), from the 1005 capture's DUR ruler",
               "gemm": "w2 only: w1's unit (S 56) is the parent's, a SEEN replicate, printed",
               "fit": "floor_estimator.joint_theta(DUR / u, q, frac), DUR = gpu__time_duration x 1005 MHz / 1e3, over the registered floor-bound w2 cells of the 1005 capture",
               "cells_w2": [c["n"] for c in w2], "cells_w1_printed": [c["n"] for c in w1], "u_w2": u2,
               "M_band": [0.98, 1.02], "F0_rival_b": r6(b_f0), "F0_band": [r6(b_f0 - 0.04), r6(b_f0 + 0.04)],
               "rule": "M HOLDS when b lies in [0.98, 1.02] (F0 FALSIFIED unless inside its own band); F0 HOLDS when b lies in [b_F0 - 0.04, b_F0 + 0.04]; else NEITHER. w1 printed with its q-only slope (ceil reads 1 / mean waves share, fluid 1.000)",
               "label": "BLIND"},
        "E5": {"what": "M against M+Z on E1's cells: rms(M), rms(M+Z), their difference",
               "status": "PRINTED, NOT A TEST (the review, section 6 E-6): M+Z - M is a near-uniform -9.6 us a call, which a board-level T0 offset reproduces; part A carries the form question",
               "illustrative_rel_shift_on_population": {"min": min(m_vs_mz), "max": max(m_vs_mz)},
               "sigma": "printed with sigma_rms = sqrt(2 sigma_page^2 / N + sigma_board^2)",
               "M+Z_refit": "T0 %.5f -> %.5f ms; c, bw, s_small, s_block unchanged" % (p0["T0"], pz["T0"])},
        "E6": {"what": "the dead-CTA term in isolation: counted SHARED - NATIVE in-kernel time per GEMM on the target's byte pages, sum over the GEMM of recorded sm__cycles_elapsed.avg (SHARED) - (NATIVE) over the 1710 MHz lock, per (G, n)",
               "cells": "G in (3, 8, 32), n = 2..9, each byte page whose V-gates pass (CLEAN) or every page (ALL)",
               "statistic": "per GEMM, the median over cells",
               "predicted": dead,
               "hypotheses": {"DEAD": "M's term: dead x DEAD_CTA_NS less one effective lifetime k_w occ_g S'_g c, so w2 at S' 7.85 hides 3.2 us of 41.7",
                              "FIXED": "the exposure does not shrink with the CTA's length: w2 reads the parent's counted SHARED - NATIVE w2 time at the same 31,248 dead CTAs (SEEN)"},
               "parent_counted_SEEN_us": parent,
               "bands_w2": {"DEAD": [r6(0.8 * dead["w2"]["pred_us"]), r6(1.2 * dead["w2"]["pred_us"])],
                            "FIXED": [r6(0.8 * parent["w2"]), r6(1.2 * parent["w2"])]},
               "band_basis": "+-20%: DEAD_CTA_NS's own se (0.58 of 8.74 us, 6.6%) and the term's miss on the parent (25.1 printed against 21.5 counted, 14%), SEEN",
               "rule": "on w2 (the primary): a hypothesis HOLDS when the median lies in its band and FALSIFIED outside it; DEAD is SELECTED when it holds and FIXED is falsified, and the reverse; else INCONCLUSIVE. w1 (5,580 against 620 dead CTAs, all hidden: predicted 0) is printed",
               "label": "BLIND (the parent's counted value is SEEN)"},
        "gates": {"timed": "every VALIDITY gate except V5 gates CLEAN (V5 reads the declaration's machinery from slopes a host-bound n = 1 bends; Granite-3B lost every timed page to it); V1 makes a page unusable in both views; a V8 that reads UNKNOWN on the eager fallback latches the page INVALID: CLEAN drops it and ALL keeps it (fixed now, the review section 5)",
                  "bytes": "every V-gate gates CLEAN for E3, E3s and E6, V10 (lock_gate.check_lock) included; V1 unusable in both. DECIDED 2026-10-05 BEFORE ANY RENTAL-3 PAGE (independent check of the build): an earlier draft exempted V10; that is withdrawn: E6 converts cycles to microseconds at 1710 MHz, so it needs the clock gate, and the stored V10 FAILs on published pages are a stale fit (all 101 published lock pages pass when lock_gate re-gates them on the nvidia-smi bracket, and five rental-2 TP-shard byte pages passed V10 as stored)",
                  "floor": "the floorlaw registration's lock-in-force check (E4 reads the 1005 capture)",
                  "views": "every verdict on ALL and CLEAN, rental 2's rule 3"},
        "ambiguities_resolved": {
            "E-1 ALL vs CLEAN": "a cell's T is the median of ms_p50 over the pages the view counts; ALL counts every page of the unit's directory with V1 passing, CLEAN every page with no failing VALIDITY gate but V5",
            "E-2 rep8": "rep8's page is in its own directory (gaps-<card>-qwen2-57b-a14b-tp8-rep8) and is never read by E1 to E6; it is R's input only",
            "E-3 population": "31 core cells, fixed above; deep cells printed with their own verdict",
            "E-4 E3": "54 cells, rms <= 3% and at most 10% beyond 5%; n = 1..9",
            "E-5 E4": "w2 only, the joint fit (identifiable: corr(q, frac) %.3f on the w2 cells); w1 printed" % (C3.corr([c["q"] for c in w2], [c["frac"] for c in w2]) or 0.0),
            "E-6 E5": "printed only",
            "E-7 UNRESOLVED": "either margin (rms to 2%, max to 5%) under the resolution",
            "E-8 above-lock": "timed cells are not dropped by clock: R3's own clock gates decide a page (CLEAN), and each cell's sm_clock_load_mhz is printed",
            "E-9 M+Z": "the source pages and the refit numbers are in source.pages and timing_params_MZ",
            "E-10 E6": "added, with the FIXED rival from the parent's counted value",
        },
        "seen_data": seen,
    }


def parent_dead_counted() -> dict:
    """The parent Qwen2-57B's counted SHARED - NATIVE in-kernel time per GEMM (SEEN):
    median over its 1710 counter pages at G in (3, 8, 32) and n = 2..9."""
    out = {}
    vals = {"w1": [], "w2": []}
    for G in BYTE_G:
        f = sorted(PARENT.glob(f"*r3-counters/lock1710/r3c-g{G}.json"))
        assert len(f) == 1, f
        d = json.loads(f[0].read_text())
        cells = {(c["arm"], c["n"]): c for c in d["cells"]}
        for g in vals:
            for n in range(2, 10):
                sh, na = cells[("shared", n)], cells[("native", n)]
                vals[g].append((sh["recorded"][g]["sm__cycles_elapsed.avg"]
                                - na["recorded"][g]["sm__cycles_elapsed.avg"]) / 1710.0)
    for g, v in vals.items():
        out[g] = r6(st.median(v))
    return out


# --------------------------------------------------------------------------
# part B: the floor law, CLOCK / CEIL / FLUID
# --------------------------------------------------------------------------

def seen_rulers() -> list[dict]:
    """SEEN (diagnosis): rental 2's floor captures on four rulers and the two
    differences, the joint fit per GEMM and capture, on the rental-2 const
    registration's floor-bound cells."""
    import floor_estimator as FE
    reg = json.loads((REG / "2026-10-01-rental2-const-gh200.json").read_text())["cells_registered"]
    out = []
    for f in sorted(R2S.glob("*floor*-r3-counters/r3f-g64*.json")):
        d = json.loads(f.read_text())
        model = d["plan"]["model"]
        lock = (d.get("clock") or {}).get("lock_mhz")
        for g in ("w1", "w2"):
            rc = [c for c in reg[model] if c["gemm"] == g and c["floor_bound"]]
            rows = C3.floor_rows(d, rc, g, None)
            if len(rows) < 5 or any(r["ACT.max"] is None for r in rows):
                continue
            if lock:
                f_r = float(lock)
            else:
                long_ = [r["EL.max"] / r["duration_ns"] * 1e3 for r in rows if r["n"] >= 10]
                f_r = float(st.median(long_))
            rows = C3.floor_rows(d, rc, g, f_r)
            q, fr, u = [r["q"] for r in rows], [r["frac"] for r in rows], rows[0]["u"]
            ent = {"capture": f"{model} {capture_key(f)}", "gemm": g, "f_ruler": r6(f_r), "u": u,
                   "cells": [r["n"] for r in rows]}
            for name in ("ACT.max", "DUR", "EL.avg", "X", "GAP"):
                jt = FE.joint_theta([r[name] / u for r in rows], q, fr)
                ent[name] = {"theta": r6(jt["theta"]), "se": r6(jt["se_theta"]), "b": r6(jt["b"])}
            ent["X_median"] = r6(st.median([r["X"] for r in rows]))
            l2 = [r["L2_mhz"] for r in rows if r["duration_ns"] >= 5e5 and r["L2_mhz"]]
            ent["L2_mhz"] = r6(st.median(l2)) if l2 else None
            out.append(ent)
    return out


def floorlaw(cx: Ctx) -> dict:
    import numpy as np
    cells, ident = {}, {}
    for m, tr in TREADS.items():
        cells[m] = cx.cells(m, tr)
        for g in ("w1", "w2"):
            fb = [c for c in cells[m] if c["gemm"] == g and c["floor_bound"]]
            ok, why = C3.identifiable([c["q"] for c in fb], [c["frac"] for c in fb], **IDENT)
            ident[f"{m} {g}"] = {"identifiable": ok, "why": why, "cells": [c["n"] for c in fb]}
    seen = seen_rulers()
    th_act = [e["ACT.max"]["theta"] for e in seen if "lock" in e["capture"]]
    th_x = [e["X"]["theta"] for e in seen]
    # the wall-time check: T(n) = a + b n + theta W(n), W(n) = sum_g frac_g(n) u_g / f (ms)
    tp4 = cells["mixtral-8x7b-tp4"]
    W = {}
    for n in TREADS["mixtral-8x7b-tp4"]:
        W[n] = sum(c["frac"] * c["u"] for c in tp4 if c["n"] == n) / 1710e3
    ns = sorted(W)
    corr_nw = C3.corr(ns, [W[n] for n in ns])
    # part 4 re-asked: tp8 w1's registered sets at the 1005 capture's DUR
    w1f = json.loads((REG / "2026-10-01-rental2-w1floor-gh200.json").read_text())
    sets = w1f["sets"]["mixtral-8x7b-tp8 w1"]
    sig_theta = 0.28 * 450.0 / 2678.0
    return {
        "registered": f"{C3.DATE}, before any rental-3 page",
        "name": C3.NAMES["floorlaw"],
        "status": "REGISTERED BEFORE ANY PAGE",
        "design": "design-r3/DESIGN.md section 3, RE-REGISTERED as the review directs (REVIEW.md section 2): the X mechanism is refuted on SEEN data and dropped; CLOCK / CEIL / FLUID; B0 added; identifiable GEMMs only; tp2 printed; the tp4 1005 capture and a wall-time check outside ncu added",
        "tool": "scripts/floor_estimator.py (joint_theta, rulers); scorer scripts/scoring/rental3/score_floorlaw.py",
        "framing": "with the wall-time check (B6) scored, part B asks whether the clock dependence of the floor law is ncu's or the kernel's; without it (unit dropped or NOT SCORED) part B is registered as the clock dependence of the NCU-MEASURED floor, and says so",
        "captures": {m: {lab: {"clocks": clk, "treads": TREADS[m]} for lab, clk in caps.items()}
                     for m, caps in CAPTURES.items()},
        "capture_path": "<date>-nvidia_gh200_480gb-<model>-<label>-r3-counters/r3f-g64[-lock<F>].json",
        "wall_unit": "<date>-nvidia_gh200_480gb-launch-floor-wall/mixtral-8x7b-tp4/cells.csv (launch_floor.py, NATIVE, G = 64, modes E240 and GR, the 1710 lock, no profiler)",
        "cells_registered": cells,
        "definitions": {
            "q": "N_live / 132", "frac": "ceil(q) - q, the last wave's idle-SM share",
            "u": "S c + F_g (c 344.1, F_w1 520, F_w2 979: CAL, 0f77622)",
            "rulers": "floor_estimator.rulers: ACT.max sm__cycles_active.max; EL.avg sm__cycles_elapsed.avg; DUR gpu__time_duration x f / 1e3 with f the LOCK of a lock capture (a base capture: f the median over its cells of n >= 10 of EL.max / duration, printed only); X = DUR - EL.avg; GAP = EL.avg - ACT.max",
            "fit": "floor_estimator.joint_theta(y / u, q, frac) per ruler, GEMM and capture: theta 1 the ceil law on that ruler, 0 the fluid law",
            "cells": "the registered floor-bound cells (CORES floor share >= 0.97 at sigma 1 and waves >= 2, cells_registered) present on the capture",
        },
        "identifiability": {"rule": f"a GEMM capture is fitted only when it has >= {IDENT['min_cells']} cells and |corr(q, frac)| <= {IDENT['max_corr']}; else NOT SCORED (printed)",
                            "per_gemm": ident,
                            "note": "computed here on the registered cells with q = N_live / 132: every new GEMM is identifiable (qwen2-tp8 w1 q = 4.85 n, granite-1B q = 3.88 n spread frac); the review's 'frac proportional to q' read q per occupancy wave (0.970 n), not the floor law's q, and the rule stands as written",
                            "se_rule": f"a GEMM capture whose se(theta) on the ruler a test reads exceeds {SE_THETA_MAX} is printed for that test, not counted"},
        "hypotheses": {
            "CLOCK": "the wall-time floor is ceil at 1005 and flatter at 1710, and the flattening sits in GAP = EL.avg - ACT.max (both SM-clock counters)",
            "CEIL": "theta about 1 on every ruler at every clock: the SEEN 1710 lows were noise",
            "FLUID": "theta about 0 at both clocks",
        },
        "tests": {
            "B0": {"what": "theta(X / u) at 1710", "gemms": "every identifiable new 1710 GEMM: granite-1B w1, w2; qwen2-tp8 w1; tp4 w1, w2 (floor and floorrep)",
                   "band": [-0.15, 0.15], "rule": "HOLDS (X is uncorrelated with frac: it sets the level, not the shape) when every scored GEMM's theta lies in the band, FALSIFIED when any lies outside",
                   "SEEN": {"theta_X_range": [min(th_x), max(th_x)], "label": "SEEN: rental 2's 14 GEMM captures"}},
            "B1": {"what": "theta_ACT.max", "captures": "every lock capture (1005, 1410, 1710) of an identifiable GEMM; base printed", "band": [0.70, 1.05],
                   "rule": "holds on a capture when theta lies in the band; B1 HOLDS when it holds on >= 80% of the scored lock captures",
                   "SEEN": {"range": [min(th_act), max(th_act)], "label": "SEEN rental-2 lock captures"}},
            "B2": {"what": "theta_DUR at 1005", "gemms": "qwen2-tp8 w1, w2; granite-1B w1, w2; tp4 w1, w2; tp8 w1, w2; tp2 w1, w2 (every identifiable 1005 GEMM)",
                   "band": [0.70, 1.05], "rule": "holds per GEMM inside the band", "SEEN": "tp4 1005: 0.95 / 0.81 (rental 2)"},
            "B3": {"what": "dtheta = theta_DUR(1005) - theta_DUR(1710), same GEMM, same board",
                   "pairs_counted": "granite-1B w1, w2; tp4 w1, w2 (floor1005 against the floor unit's 1710)",
                   "pairs_printed": "qwen2-tp8 w2 (its 1710 DUR is printed: X's cell-to-cell scatter there is 1 to 2 u); tp8 and tp2 (their 1710 capture is rental 2's, another board)",
                   "rule": "per pair where theta_DUR(1710) <= 0.70: CLOCK predicts dtheta >= 0.25, CEIL and FLUID |dtheta| <= 0.20"},
            "B4": {"what": "theta_DUR at 1710", "gemms": "granite-1B w1, w2; tp4 w1, w2; qwen2-tp8 w1 (new 1710 GEMMs; qwen2-tp8 w2 printed)",
                   "rule": "CLOCK: theta_DUR <= theta_ACT.max - 0.20 on at least 2 of them, at least one of them a BLIND shape (granite-1B or qwen2-tp8; DECIDED 2026-10-05 BEFORE ANY RENTAL-3 PAGE (independent check of the build): the overlap narrowing below was chosen on SEEN rental-2 values, so the tp4 replicate alone, whose SEEN values meet the condition, may not pass the clause); CEIL: theta_DUR >= 0.70 on every one; a GEMM whose theta_DUR lies where the two clauses' bands overlap (theta_ACT.max > 0.90 and 0.70 <= theta_DUR <= theta_ACT.max - 0.20) is printed and enters neither clause (the design's 'touch' rule, narrowed to the overlap so a theta_ACT.max above 0.90 alone does not drop a GEMM: SEEN theta_ACT.max reads 0.91 to 0.95 on three of rental 2's 1710 GEMMs)"},
            "B5": {"what": "X at 1005 (median of cells)", "status": "PRINTED (a lock-in-force reading, in no verdict; SEEN 312 to 342 cycles)"},
            "B6": {"what": "the wall-time floor OUTSIDE ncu: GR (graph replay, the call's GPU time) cells of the wall unit, T(n) = a + b n + theta_wall W(n), W(n) = sum_g frac_g(n) u_g / 1710 MHz (ms), OLS over the unit's treads (every tp4 tread is floor-bound on both GEMMs at G = 64)",
                   "W_ms": {str(n): r6(W[n]) for n in ns}, "corr_n_W": r6(corr_nw),
                   "rule": "theta_wall >= 0.70: the 1710 flattening is ncu's (CLOCK reads 'in the ncu replay only'); theta_wall <= 0.50: it is the kernel's (CLOCK reads 'in wall time'); between: INCONCLUSIVE; se(theta_wall) > 0.20 or |corr(n, W)| > 0.95: NOT SCORED. E240 is printed beside GR",
                   "label": "BLIND: no wall-time page at G = 64 exists"},
            "B7": {"what": "part 4 re-asked on the clean ruler: tp8 w1's registered sets on the 1005 capture's DUR", "sets": sets["sets"],
                   "asymptote": sets["asymptote"], "sigma_theta": r6(sig_theta),
                   "rule": "theta and delta by floor_estimator.theta_delta over the three sets (slopes by implied_slope on DUR); H_EST HOLDS when theta > 0.5 and |theta - 1| <= 0.20 (2 sigma_theta, inflated for the SEEN 0.81 to 0.95 at 1005); FLUID HOLDS when |theta| <= 0.20; else INCONCLUSIVE; delta printed, never scored"},
        },
        "verdict": {
            "CLOCK": "HOLDS when B1 HOLDS, B2 holds on every scored 1005 GEMM, B3 gives dtheta >= 0.25 on every counted pair where it applies, B4's CLOCK clause holds and B0 HOLDS; FALSIFIED when B2 fails on 2 or more GEMMs or B4's CEIL clause holds",
            "CEIL": "HOLDS when B4's CEIL clause holds and B2 holds on every scored 1005 GEMM; FALSIFIED when B4's CEIL clause fails",
            "FLUID": "FALSIFIED when B1 HOLDS or B2 holds on 2 or more GEMMs; HOLDS when theta_DUR <= 0.30 on every scored 1005 and 1710 GEMM",
            "selected": "a hypothesis is SELECTED when it HOLDS and the other two are FALSIFIED; else INCONCLUSIVE. B6 then labels CLOCK 'in wall time' or 'in the ncu replay only', and its absence labels the claim 'ncu-measured'",
        },
        "X_clock_law": {"status": "PRINTED (3.5 kept as printed, the review)",
                        "X1710": "per GEMM, the median over the GEMM's registered cells of X at the tp4 floor unit's 1710 capture",
                        "X1005": "the same at the tp4 floor1005 capture", "X1410": "the same at floor1410",
                        "LIN": "X1410 within +-35% of X1005 + (405 / 705)(X1710 - X1005)", "HIGH": "X1410 <= 0.25 X1710",
                        "FLAT": "X1410 >= 0.80 X1710", "base": "the base capture's X is printed (ncu's pin is not an -lgc lock)"},
        "gates": {"floor": {"null_tol": 0.03, "l2_lock_mhz": 1710.0, "l2_band_mhz": [1690.0, 1715.0], "l2_min_duration_ns": 500000.0,
                            "smi_tol_mhz": 15.0,
                            "rule": "a capture under a lock is in CLEAN when V1 passes and the lock is in force: the null kernel's median clock within 3% of the lock AND, at the 1710 lock only, the median over cells of duration >= 0.5 ms of lts__cycles_elapsed.avg / duration in [1690, 1715] MHz (SEEN 1700.5 to 1706.0 on every rental-2 1710 and base cell of 0.5 ms or more; DECIDED 2026-10-05 BEFORE ANY RENTAL-3 PAGE (independent check of the build): when fewer than 3 cells reach 0.5 ms, as on Granite-1B (30 to 230 us), the L2 leg is PRINTED, not gated, and the null kernel and nvidia-smi still gate: short cells read low through a fixed offset of about 0.75 us, 1675 to 1677 MHz under 50 us, 1694.5 to 1700 at 150 to 250 us, so an all-cell median would put a held lock on a coin flip; the L2 reads 1125 MHz at a 1005 lock, it follows a low SM lock, so 1005 and 1410 print it) AND nvidia-smi before and after within 15 MHz (one step) of the lock. FL1 is printed and gates nothing. Rental 2's rule 2 (a cell over the lock by more than 15 MHz is dropped) is kept; base captures carry no lock and no check",
                            "SEEN_L2": [e["L2_mhz"] for e in seen if e.get("L2_mhz")]},
                  "wall": "the wall unit has no validity gate; its manifest's profiler_enabled_ever must be False (else NOT SCORED)"},
        "ambiguities_resolved": {
            "B-1 cells": "the registered floor-bound cells (CORES share >= 0.97, waves >= 2), cells_registered; never the route's 4-wave set",
            "B-2 singular": "the identifiability rule above (>= 5 cells, |corr(q, frac)| <= 0.95)",
            "B-3 B1 denominator": "lock captures only (1005, 1410, 1710); base printed",
            "B-4 points or intervals": "point estimates against the registered bands; a GEMM capture whose se(theta) exceeds 0.15 is printed for that test",
            "B-5 B5": "printed only",
            "B-6 FLUID and B1": "FLUID is FALSIFIED by B1 HOLDING (>= 80% of lock captures in [0.70, 1.05]) or by B2 holding on 2 or more GEMMs",
            "B-7 X1710": "per GEMM median over that GEMM's cells",
        },
        "consequence_for_E": "M prices the floor with ceil(q) at 1710. If CLOCK holds in wall time, M's floor-bound w1 cells at large frac are mispriced; qwen2-tp8 w1's frac is 0.06 to 0.27 at the scored n (q = 4.85 n), so the effect is small there; printed with E",
        "seen_rulers": seen,
        "seen_data": ["every band is read off rental 2's captures (SEEN, labelled in each test); the cells are new captures",
                      "tp8's and tp2's 1710 captures are rental 2's (another board): their B3 pairs are printed, not counted",
                      "the X mechanism of the design (X flattens theta) is refuted on the SEEN theta(X/u) above; the drop sits in GAP"],
    }


# --------------------------------------------------------------------------
# part A: the per-GEMM constant's form
# --------------------------------------------------------------------------

def zform(cx: Ctx, series: list[dict], fits: dict) -> dict:
    import floor_estimator as FE
    import r3_timing_model as TM
    aff = fits["AFF"]["coef"]
    # the timing model carries the AFF coefficients: they must be this fit's
    assert abs(aff[0] - TM.GEMM_CONST_CYCLES) < 1.0, (aff, TM.GEMM_CONST_CYCLES)
    assert abs(aff[1] - TM.GEMM_CONST_PER_U) < 5e-5, (aff, TM.GEMM_CONST_PER_U)
    seen = [r for r in series if not r["cal"]]
    resid = []
    for r in seen:
        resid.append({"series": f"{r['model']} {r['session'][:10]} {r['capture']} {r['gemm']}", "u": r["u"],
                      "mhz": r["mhz"], "Z": r["Z"], "sigma": r6(z_sigma(r)),
                      **{k: r6(r["Z"] - z_pred(fits, k, r["u"], r["mhz"])) for k in FORMS}})
    rms = {k: r6(math.sqrt(st.mean([x[k] ** 2 for x in resid]))) for k in FORMS}
    # SEEN priors: rental 2's same-board clock ratios (pooling fixed: each capture alone)
    r2 = [r for r in series if r["session"].startswith("2026-10-02")]
    priors = []
    for r in r2:
        if r["capture"] != "lock1710":
            continue
        for other in ("base", "lock1005"):
            o = next((x for x in r2 if x["model"] == r["model"] and x["gemm"] == r["gemm"]
                      and x["capture"] == other), None)
            if o:
                rr = o["Z"] / r["Z"]
                sig = rr * math.sqrt((z_sigma(o) / o["Z"]) ** 2 + (z_sigma(r) / r["Z"]) ** 2)
                priors.append({"model": r["model"], "gemm": r["gemm"], "pair": f"{other}/lock1710",
                               "r": r6(rr), "sigma_r_registered_noise": r6(sig),
                               "f_ratio": r6(o["mhz"] / r["mhz"]),
                               "AFF": 1.0, "MIX": r6(z_pred(fits, "MIX", r["u"], o["mhz"]) / z_pred(fits, "MIX", r["u"], r["mhz"]))})
    # the separating GEMMs, their cells and noise
    sep = {}
    for m, g, role in ((TARGET, "w2", "scored"), ("granite-3.0-1b-a400m", "w2", "scored"),
                       ("granite-3.0-1b-a400m", "w1", "scored"), (TARGET, "w1", "printed")):
        cs = [c for c in cx.cells(m, TREADS[m]) if c["gemm"] == g and c["floor_bound"]]
        u = cs[0]["u"]
        q = [c["q"] for c in cs]
        s_int = FE.intercept_sigma(q, max(250.0, 0.02 * u))
        s_tot = math.sqrt(s_int ** 2 + 300.0 ** 2 + 450.0 ** 2)
        lo = [c["q"] for c in cs if c["n"] <= 8]
        hi = [c["q"] for c in cs if c["n"] >= 8]
        cent = {k: r6(z_pred(fits, k, u, 1690.0)) for k in ("PROP", "AFF", "MIX")}
        sep[f"{m} {g}"] = {
            "role": role, "u": u, "cells": [c["n"] for c in cs], "sigma_int": r6(s_int), "sigma_tot": r6(s_tot),
            "A1_centres_at_1690": cent,
            "A1_bands_at_1690": {k: [r6(v - 2 * s_tot), r6(v + 2 * s_tot)] for k, v in cent.items()},
            "A1_forms": (["PROP", "MIX"] if (m, g) == ("granite-3.0-1b-a400m", "w1") else ["PROP", "AFF", "MIX"]),
            "A2_primary_MIX_ratio_base1375_over_1690": r6(z_pred(fits, "MIX", u, 1375.0) / z_pred(fits, "MIX", u, 1690.0)),
            "A2_second_MIX_ratio_1005_over_1690": r6(z_pred(fits, "MIX", u, 1005.0) / z_pred(fits, "MIX", u, 1690.0)),
            "half_fits": {"n<=8": [c["n"] for c in cs if c["n"] <= 8], "n>=8": [c["n"] for c in cs if c["n"] >= 8],
                          "sigma_lo": r6(FE.intercept_sigma(lo, max(250.0, 0.02 * u))) if len(lo) >= 3 else None,
                          "sigma_hi": r6(FE.intercept_sigma(hi, max(250.0, 0.02 * u))) if len(hi) >= 3 else None},
            "extrapolation_waves": r6(min(q))}
        zc = z_pred(fits, "AFF", u, 1690.0)
        sr = math.sqrt(2.0) * s_int / zc
        for lever, f in (("base", 1375.0), ("lock1005", 1005.0)):
            rm = z_pred(fits, "MIX", u, f) / z_pred(fits, "MIX", u, 1690.0)
            sep[f"{m} {g}"].setdefault("power", {})[lever] = {
                "gap_AFF_MIX": r6(1 - rm), "two_sigma_r": r6(2 * sr),
                "MID_window": [r6(rm + 2 * sr), r6(1 - 2 * sr)],
                "MID_possible": bool(rm + 2 * sr < 1 - 2 * sr)}
    base_clocks = [r["mhz"] for r in r2 if r["capture"] == "base"]
    return {
        "registered": f"{C3.DATE}, before any rental-3 page",
        "name": C3.NAMES["zform"],
        "status": "REGISTERED BEFORE ANY PAGE",
        "design": "design-r3/DESIGN.md section 4 with REVIEW.md section 3: the pooling bug fixed, the SEEN tp4 ratios registered as priors, MID added, base / 1710 the primary clock lever, the curvature check, the verdict arithmetic stated",
        "tool": "scorer scripts/scoring/rental3/score_zform.py; the AFF coefficients are r3_timing_model's GEMM_CONST (asserted equal here)",
        "Z": "floor_estimator.intercept of sm__cycles_active.avg on q over a capture's registered floor-bound cells of one GEMM, the slope free (rental 2's score_const.z_intercept); one Z per (capture FILE, GEMM): base, lock1710, lock1005 never pooled",
        "fits_CAL": fits,
        "fit_basis": "weighted least squares on the CAL models' G = 64 base and 1710-lock captures (8x7B 09-27 and 09-30, 8x22B, Qwen2-57B, OLMoE), weight 1 / sigma_Z^2, sigma_Z = max(300, the intercept's error at sigma_active = max(250, 0.02 u)); each series at its own measured clock (median sm_clock_mhz of its cells)",
        "forms": {"PROP": "Z = zeta u (one zeta; the form question (a) names)", "AFF": "Z = g + b u, SM cycles, clock-free (the cycle form; r3_timing_model --gemm-const)",
                  "MIX": "Z = a (f / 1710) + b u: a constant in ns plus a cycle term", "MIX3": "a (f / 1710) + g + b u: PRINTED, never required to be FALSIFIED (it nests AFF and MIX)"},
        "seen_residuals": {"per_series": resid, "rms": rms, "series": len(resid),
                           "label": "SEEN: every non-CAL published capture, each capture file its own series (the design pooled tp4's 1710 and 1005 cells; fixed)"},
        "seen_priors": {"pairs": priors, "label": "SEEN, registered as labelled priors: rental 2's same-board ratios sit between AFF (1.0) and MIX; the likely outcome on the new GEMMs is INCONCLUSIVE on the primary lever (MID unreachable at the registered noise, A2.power); MID_1005_printed reports the 1005 lever (DECIDED 2026-10-05 BEFORE ANY RENTAL-3 PAGE (independent check of the build): the label said MID)"},
        "separating_gemms": sep,
        "noise": {"sigma_int": "max(the registered sigma_int, the realised intercept standard error of the fit) per capture", "sigma_board": 300.0,
                  "sigma_model": 450.0, "basis": "sigma_board: the same shape on two boards (tp8 r1 / r2 dZ 123 to 467, JetMoE 09-29 / 09-30 264 to 277, SEEN); sigma_model: the SEEN rms of AFF and MIX at u < 12k"},
        "A1": {"what": "Z at the 1710 capture against each form's centre at the capture's MEASURED clock (median sm_clock_mhz of its cells)",
               "band": "centre +- 2 sigma_tot, sigma_tot = sqrt(sigma_int^2 + sigma_board^2 + sigma_model^2)",
               "forms_per_gemm": "PROP, AFF and MIX on qwen2-tp8 w2 and granite-1B w2; PROP and MIX on granite-1B w1 (PROP and AFF overlap there); qwen2-tp8 w1 printed"},
        "A2": {"primary": "r = Z_base / Z_1710, same board: the SM clock moves and the L2 stays at about 1704 MHz",
               "second": "r' = Z_1005 / Z_1710, PRINTED with the label 'SM and L2 moved together' (the L2 follows a 1005 lock to 1125 MHz)",
               "predictions": "AFF and PROP: 1.0; MIX: Z_MIX(u, f_base) / Z_MIX(u, f_1710) at the two captures' measured clocks",
               "sigma_r": "r sqrt(sigma_int,base^2 + sigma_int,1710^2) / Z_1710 (sigma_board cancels on one board)",
               "rule": "AFF and PROP FAIL A2 when |r - 1| > 2 sigma_r; MIX FAILS when |r - r_MIX| > 2 sigma_r; MID when r_MIX + 2 sigma_r < r < 1 - 2 sigma_r ('neither cycle-constant nor ns-constant')",
               "power": "per GEMM in separating_gemms.power, at the registered sigma_int: on the base / 1710 lever the AFF-MIX gap (0.14 to 0.16) exceeds 2 sigma_r (0.13 to 0.15) by little, so an endpoint reading falsifies the other form only narrowly and the MID window is EMPTY: MID cannot be read on the primary lever. It is read on the printed 1005 lever (gap 0.30 to 0.35, window open) as MID_1005_printed; a reading between the forms on the primary lever is INCONCLUSIVE, which the SEEN priors (0.92 to 0.97) make the likely outcome",
               "SEEN_base_clocks_mhz": base_clocks},
        "curvature": "per GEMM and capture, the intercepts of the n <= 8 and n >= 8 half-fits agree within 2 sqrt(sigma_lo^2 + sigma_hi^2), else that GEMM is NOT SCORED (qwen2-tp8 w2 extrapolates 13.6 waves to q = 0 and Z is 1.26 u)",
        "verdict": {"per_form": "a form FAILS a GEMM when A1 (where it is scored for it) or A2 fails there; it is FALSIFIED when it fails on at least 2 of the GEMMs scored for it (AFF: A1 on 2 GEMMs and A2 on 3 count as 3 GEMMs)",
                    "selected": "SELECTED when it fails on no scored GEMM and every other one of PROP, AFF, MIX is FALSIFIED; MIX3 is printed and need not be FALSIFIED",
                    "MID": "when A2 reads MID on 2 or more GEMMs, the verdict is MID (unreachable on the primary lever at the registered noise: see A2.power); MID_1005_printed reports the 1005 lever's reading beside it",
                    "else": "INCONCLUSIVE, each A printed"},
        "also_printed": ["the Z / u of every new capture", "the base capture's Z, a third clock", "K3's tp4 value refit under each form (the tp4 floor and floor1005 units)"],
        "ambiguities_resolved": {
            "A-1 sigma_int": "the larger of the registered and the realised intercept error",
            "A-2 cell set": "the registered floor-bound cells (floorlaw cells_registered)",
            "A-3 MIX's clock": "each capture's measured clock",
            "A-4 2 of 3": "per form, at least 2 of the GEMMs scored for it",
            "A-5 MIX3": "printed only",
            "A-6 pooling": "one series per capture file",
            "A-7 A2's primary": "base / 1710",
        },
        "leakage": "the coefficients are CAL; sigma_board and sigma_model are SEEN, labelled; the cells are BLIND",
        "seen_data": ["the SEEN residuals and priors are diagnosis, never a test", "Z's form at small u was diagnosed on SEEN Granite-3B and tp8 w2 series"],
    }


# --------------------------------------------------------------------------
# part C: which feature lifts G = 1 survival
# --------------------------------------------------------------------------

PAGES_C = {"k32s8": {"block_k": 32, "num_stages": 8}, "k128s4": {"block_k": 128, "num_stages": 4},
           "k128s3": {"block_k": 128, "num_stages": 3}, "k64s7": {"block_k": 64, "num_stages": 7},
           "k64s3": {"block_k": 64, "num_stages": 3}}
EXPECTED_CTAS = {"k32s8": 4, "k128s4": 2, "k128s3": 3, "k64s7": 2, "k64s3": 4}


def classify(h: str, ctas: int, stages: int, bk: int) -> str:
    occ, depth, width = ctas <= 2, stages >= 7, bk >= 128
    lifted = {"OCC": occ, "DEPTH": depth, "WIDTH": width, "U-OCC": occ or width,
              "U-DEPTH": depth or width, "NULL": False}[h]
    return "lifted" if lifted else "base"


HYPS_C = ("OCC", "DEPTH", "WIDTH", "U-OCC", "U-DEPTH", "NULL")


def stages() -> dict:
    import l2_survival as L
    pats = {h: {p: classify(h, EXPECTED_CTAS[p], v["num_stages"], v["block_k"]) for p, v in PAGES_C.items()}
            for h in HYPS_C}
    pairs = {}
    for i, a in enumerate(HYPS_C):
        for b in HYPS_C[i + 1:]:
            diff = [p for p in PAGES_C if pats[a][p] != pats[b][p]]
            pairs[f"{a}/{b}"] = diff
    base_page = sorted(R2S.glob(f"*-{C3.CARD}-mixtral-8x7b-tp2-l2base-r3-counters/lock1710/r3c-g1.json"))
    assert len(base_page) == 1
    sv = L.survival(json.loads(base_page[0].read_text()))
    s_r2 = {str(n): r6(v) for n, v in sv["w2"]["s"].items()}
    return {
        "registered": f"{C3.DATE}, before any rental-3 page",
        "name": C3.NAMES["stages"],
        "status": "REGISTERED BEFORE ANY PAGE",
        "design": "design-r3/DESIGN.md section 5 with REVIEW.md section 4: the one-page pairs, NULL's role, k128s3's occupancy verified on the page, drop-group c",
        "tool": "scripts/l2_survival.py, wave_split_bytes.page_window; scorer scripts/scoring/rental3/score_stages.py",
        "pages": {"path": "<date>-nvidia_gh200_480gb-mixtral-8x7b-tp2-<label>-r3-counters/lock1710/r3c-g1.json",
                  "base": "l2base", "knobs": PAGES_C, "control": "k64s3", "G": 1},
        "gemm": "w2 (x 0.92, P 64) is classified; w1 (x 1.86, P 224) is printed",
        "occupancy": {"field": "wave_split_bytes.page_window(page, 'w2')[1]: the least of the four recorded launch__occupancy_limit_* of w2, the same in every cell",
                      "expected": EXPECTED_CTAS,
                      "expected_basis": "the design's estimate from shared memory (192 BK bytes a stage, 228 KiB an SM, 1 KiB a CTA) capped by the SEEN register limit; never used to score: each page's column is re-keyed from its RECORDED value",
                      "k128s3": "scored as k128s3 only when its recorded w2 occupancy is 3; otherwise its column is re-keyed and the pairs it alone separates (OCC / U-OCC) read NO SEPARATION"},
        "stages_and_block_k": "read off each page's design block (num_stages, block_k), asserted equal to the plan's",
        "hypotheses": {"OCC": "lifted when CTAs per SM <= 2", "DEPTH": "lifted when num_stages >= 7 (SEEN puts it between 6 and 8)",
                       "WIDTH": "lifted when BLOCK_K >= 128", "U-OCC": "OCC or WIDTH", "U-DEPTH": "DEPTH or WIDTH",
                       "NULL": "never lifted (the knob does nothing): the control's prediction, SEEN-refuted on s8 and BK128 and still scored"},
        "patterns_at_expected": pats,
        "pairs": {"differ_on": pairs,
                  "one_page_pairs": {k: v for k, v in pairs.items() if len(v) == 1},
                  "rule": "a pair separated by ONE page needs that page classified; else that pair reads NO SEPARATION, and every other pair is still scored"},
        "bands": {"lifted": [0.10, 0.35], "base": [-0.06, 0.06], "ns": list(range(4, 10)), "need": 4,
                  "ds": "s_knob(n) - s_base(n), the same-board l2base page",
                  "basis": "SEEN tp2 w2: lifted pages +0.13 to +0.28 per n (s8 +0.18 to +0.28, BK128/s2 +0.13 to +0.27); base pages (s6, BK32, pad 7) -0.03 to +0.03; same-board noise 0.02",
                  "rule": "a page is lifted (base) when at least 4 of n = 4..9 fall in the lifted (base) band; else unclassified"},
        "verdict": {"falsify": "a hypothesis is FALSIFIED by any classified page whose class it gets wrong (at that page's recorded occupancy)",
                    "select": "SELECTED when it matches every classified page, at least 4 of the 5 pages are classified, and every other hypothesis (NULL included) is FALSIFIED",
                    "control": "k64s3 is predicted base by every hypothesis: lifted there FALSIFIES all six (the lift has another cause); unclassified or missing changes nothing",
                    "missing": "a page that is missing or fails to compile is NOT SCORED and leaves the population; the 4-page minimum then decides",
                    "else": "INCONCLUSIVE, the survivors and every NO SEPARATION pair printed"},
        "controls": {"board_check": {"rule": "the l2base page's w2 s within 0.05 of rental 2's tp2 l2base at 4 of n = 4..9", "rental2_s_w2_SEEN": s_r2,
                                     "effect": "a failure turns every verdict INCONCLUSIVE"},
                     "F_over_Mn": {"band": [0.50, 0.56], "rule": "every PRIVATE cell of every page", "effect": "a failing page's class is printed and the page leaves the count"},
                     "V6": "a page whose V6 is not PASS is FLAGGED: it leaves the count", "V7_V10": "do not gate (bytes; s reads SHARED and PRIVATE)"},
        "co_prediction": {"status": "PRINTED", "rule": "the far share of SHARED's extra hits (l2_survival far, mean over n = 4..9): lifted pages <= 0.22, base pages >= 0.23; a lifted page with a base far share is printed as a mechanism counter-example",
                          "SEEN": "lifted 0.15 and 0.19 to 0.20; base 0.25 to 0.27"},
        "ambiguities_resolved": {"C-1": "w2 only; w1 printed", "C-2": "page_window's least recorded limit of w2",
                                 "C-3": "k64s3 lifted falsifies all; else no effect", "C-4": "printed", "C-5": "NULL must be FALSIFIED for a SELECTED",
                                 "C-6": "the one-page-pair rule above", "C-7": "NOT SCORED page, population shrinks, 4-of-5 minimum"},
        "seen_data": ["the bands are SEEN (rental 2's knob pages); the five knob combinations are on no page", "rental 2's tp2 l2base s is SEEN, the board check's reference"],
    }


# --------------------------------------------------------------------------
# part D: the launch floor's flush ladder
# --------------------------------------------------------------------------

def flush() -> dict:
    lf2 = json.loads((REG / "2026-10-01-rental2-launch2-gh200.json").read_text())
    creg = {arm: {n: lf2["C_reg_ms"]["mixtral-8x7b-tp8"][arm][n] for n in ("1", "2", "3")}
            for arm in ("native", "shared", "private")}
    mib = {"E0": 0, "E120": 120, "E240": 240, "E360": 360, "E480": 480}
    F = {m: r6(v * 2 ** 20 / 3.7261e12 * 1e3) for m, v in mib.items()}
    dF = r6(F["E360"] - F["E240"])
    return {
        "registered": f"{C3.DATE}, before any rental-3 page",
        "name": C3.NAMES["flush"],
        "status": "REGISTERED BEFORE ANY PAGE",
        "design": "design-r3/DESIGN.md section 6 with REVIEW.md sections 5 and 6 (D): scored on the modes of 240 MiB and above; E120 under the 4x rule",
        "tool": "scripts/launch_floor.py --phase timed / trace (E120 and E360 in MODES, FLUSH_MB and make_flushers); scorer scripts/scoring/rental3/score_flush.py",
        "unit": {"model": "mixtral-8x7b-tp8", "label": "r3", "treads": [1, 2, 3], "modes_planned": ["E0", "E240", "E360", "E480"],
                 "path": "<date>-nvidia_gh200_480gb-launch-floor-r3/mixtral-8x7b-tp8/"},
        "E120": "NOT PLANNED: 120 MiB is 2x the GH200's 60 MiB L2, under timing.flush_mb_for_device's own 4x rule, so it evicts incompletely and the kernel after it gets faster too; launch_floor.py accepts it and the scorer prints it when present, never scores it",
        "C_reg_ms": creg, "C_reg_source": "the 2026-10-01 rental-2 launch2 registration's C_reg_ms (G = 4, CAL)",
        "F_gpu_ms": F, "F_basis": "MiB x 2^20 bytes over the 3.7261 TB/s ruler (never the trace's F_tr: P3's instrument problem)",
        "definitions": {"Phi": "Phi(F) = H_cell(F) - I(F), each mode's own cell: H_cell the median over the cell's repeats of host_enqueue_ms / calls_per_burst, I the median of ms_p50 (rental 2's definitions)",
                        "qualifying": "a (arm, n) cell is scored when, at every one of E240, E360 and E480, it is host-bound by rule (C_reg + F_mode < H_pre, hostprobe.csv) AND measured host-bound (host_bound_fraction >= 0.8) AND not an edge (|C_reg + F_mode - H_pre| > 0.020 ms)"},
        "hypotheses": {"OVERLAP": "Phi(F) = F_gpu(F) - h_flush + delta(level): linear in F_gpu with slope 1 at fixed level",
                       "SAT": "Phi saturates at the E240 value"},
        "tests": {"differences": "per qualifying cell, d1 = Phi(E360) - Phi(E240) and d2 = Phi(E480) - Phi(E360) (F_gpu steps %.1f and %.1f us)" % (dF * 1e3, (F["E480"] - F["E360"]) * 1e3),
                  "OVERLAP_band_us": [r6(dF * 1e3 - 12), r6(dF * 1e3 + 12)], "SAT_band_us": [-10.0, 10.0],
                  "band_basis": "+-12 us: rental 2's P3 / P4 band (E0 - E240 is a difference of two host-paced intervals that scatter about 8 us between arms at fixed n), SEEN",
                  "cell_rule": "a cell supports a hypothesis when both d1 and d2 lie in its band (the bands are disjoint)",
                  "rule": "per model, a hypothesis is SELECTED when at least 2/3 of the qualifying cells support it; FALSIFIED when at least 2/3 support neither it nor lie in its band; else INCONCLUSIVE; NOT TESTED when fewer than 3 cells qualify",
                  "slope": "the OLS slope of Phi on F_gpu over E240, E360 and E480 per cell is printed (OVERLAP predicts 1, SAT 0)"},
        "printed": ["E0 and E120 (E0 warm, E120 partly warm: they change the kernel, so neither enters a slope)", "the two-level Phi law below"],
        "two_level_law_SEEN": {"rule": "Phi_hi (about 100 us) while C_reg < C*, Phi_lo (75 to 85 us) once C_reg > C*",
                               "C_star": "between 0.158 and 0.193 ms on tp8 and between GR(2) and GR(3) on Granite-3B",
                               "label": "SEEN-fitted (rental 2's launch.score.json), diagnosis"},
        "gates": "the launch directory records no validity gate; P0 (rental 2's) is printed: a P0 failure marks every verdict 'host drift', as rental 2 did",
        "ambiguities_resolved": {"D-1": "each mode's own H_cell and I", "D-2": "the 2/3 rule per model over qualifying cells",
                                 "D-3": "the verdict reads d1 and d2; the 3-mode slope is printed", "D-4": "rental 2's P3 / P4 band, SEEN",
                                 "D-5": "E0 and E120 printed only"},
        "seen_data": ["the band and the two-level law are SEEN (rental 2); E360 is on no page"],
    }


# --------------------------------------------------------------------------
# part R and the K1 / K4 replicate
# --------------------------------------------------------------------------

def sigma_board() -> dict:
    """SEEN: 8x7B's timed pages common to the 2026-09-25 and 2026-09-27 boards (both
    1710 lock, published): per common SHARED and PRIVATE cell |T27 / T25 - 1|."""
    import cores_heldout_predict as CP

    def cells(reports):
        d = {}
        for j in reports:
            for (arm, n), ms in C3.treads_ms(j).items():
                d.setdefault((arm, C3.group_of(j), n), []).append(ms)
        return {k: st.median(v) for k, v in d.items()}
    p25, ids25 = [], []
    for rep in sorted(S25.rglob("private_weight_reference/*/report.json")):
        if any(rep.parent.name.endswith(i) for i in S25_RUNS):
            p25.append(json.loads(rep.read_text()))
            ids25.append(rep.parent.name[-8:])
    p27 = [json.loads((p / "report.json").read_text()) for p in CP.source_pages()]
    ids27 = [p.name[-8:] for p in CP.source_pages()]
    a, b = cells(p25), cells(p27)
    com = sorted(k for k in a if k in b and k[0] in ("shared", "private"))
    d = [b[k] / a[k] - 1 for k in com]
    return {"pages_2026_09_25": ids25, "pages_2026_09_27": ids27, "cells": len(com),
            "groups": sorted({k[1] for k in com}), "rms": r6(math.sqrt(st.mean([x * x for x in d]))),
            "mean": r6(st.mean(d)), "label": "SEEN"}


def replicate(cx: Ctx) -> dict:
    sb = sigma_board()
    tc = {(c["arm"], c["G"], c["n"]): c for c in cx.pred["cells"]}
    rep_cells = [f"{arm} G=8 n={n}" for arm in ("shared", "private") for n in range(1, 7)
                 if tc[(arm, 8, n)]["ms"] + F240_MS >= HOST_MS]
    k2 = json.loads((REG / "2026-10-01-rental2-const-gh200.json").read_text())
    tp4 = cx.cells("mixtral-8x7b-tp4", TREADS["mixtral-8x7b-tp4"])
    return {
        "registered": f"{C3.DATE}, before any rental-3 page",
        "name": C3.NAMES["replicate"],
        "status": "REGISTERED BEFORE ANY PAGE: noise models, no verdict of their own",
        "design": "design-r3/DESIGN.md section 2.4 (part R) with REVIEW.md (R: name the page IDs); and the grading's finding that R did not replicate rental 2's per-GEMM-constant tests K1 and K4: the RK capture",
        "tool": "scorer scripts/scoring/rental3/score_replicate.py",
        "R": {
            "sigma_cell": "R3's own repeat spread per cell (9 repeats; the page's CI), printed",
            "sigma_page": {"what": "the rep8 unit's G = 8 page (treads 6) against the e2e timed step's G = 8 page, same board: per cell |T_rep / T_e2e - 1| over the cells that pass E's host rule; sigma_page is its rms",
                           "cells": rep_cells, "pages": "gaps-<card>-qwen2-57b-a14b-tp8-rep8 against gaps-<card>-qwen2-57b-a14b-tp8-e2e (session tag -p2-), both under the same two views as E"},
            "sigma_board": sb,
            "resolution": "sqrt(sigma_page^2 + sigma_board^2): E1's UNRESOLVED label reads it",
            "without_rep8": "when rep8 is dropped or NOT SCORED, sigma_page is not formed: E1 prints R3's per-cell CI and sigma_board alone and the label reads 'resolution: board only'",
        },
        "RK": {
            "what": "a same-board floor replicate for rental 2's K1 and K4, which no earlier capture gave a noise floor: the tp4 floorrep unit's 1710 capture against the tp4 floor unit's 1710 capture (same treads, consecutive units, one board)",
            "cells": [c["n"] for c in tp4 if c["floor_bound"] and c["gemm"] == "w1"],
            "cells_w2": [c["n"] for c in tp4 if c["floor_bound"] and c["gemm"] == "w2"],
            "statistics": {"L": "sm__cycles_elapsed.avg - sm__cycles_active.max (K1's)",
                           "D_imb": "sm__cycles_active.max - sm__cycles_active.avg - (ceil(q) - q) u (K4's)"},
            "noise": "per statistic and GEMM, sigma = rms over the registered cells of (A - B) / sqrt(2): the per-capture noise of one cell",
            "rule": "K1 (K4) is RESOLVED on a pool when its verdict would not change if every cell within 2 sigma of a threshold (K1: 5,000 cycles; K4: +-0.2 u and 0.3 u) were flipped; else UNRESOLVED. Applied to rental 2's published K1 / K4 pool (SEEN: scripts/scoring/rental2/const.score.json) and to rental 3's tp4 1710 captures; printed beside rental 2's verdicts, which stand",
            "K1_threshold": k2["K1"]["threshold_cycles"], "K4_bounds_u": [0.2, 0.3],
            "drop_order": "the floorrep unit is in drop-group x with the tp4 floor, floor1005, floor1410 and wall units: it needs the floor unit's 1710 capture (its same-board pair) and fits the budget there (2 min); when x drops, RK is NOT RUN and K1 / K4 keep no noise floor",
        },
        "seen_data": ["sigma_board is SEEN (8x7B's 09-25 and 09-27 pages)", "rental 2's K1 / K4 pool is SEEN"],
    }


# --------------------------------------------------------------------------
# the .txt renderers
# --------------------------------------------------------------------------

def _flat(prefix: str, v, out: list, depth=0):
    if isinstance(v, dict):
        for k, x in v.items():
            if isinstance(x, (dict, list)) and depth < 1:
                out.append(f"{prefix}{k}:")
                _flat(prefix + "  ", x, out, depth + 1)
            else:
                out.append(f"{prefix}{k}: {json.dumps(x, default=str) if not isinstance(x, str) else x}")
    elif isinstance(v, list):
        for x in v:
            out.append(f"{prefix}- {json.dumps(x, default=str) if not isinstance(x, str) else x}")
    else:
        out.append(f"{prefix}{v}")


SKIP_TXT = {"cells_registered", "seen_rulers", "q_pred", "q_status", "fits_CAL", "seen_residuals",
            "code_pins"}


def txt(part: str, d: dict) -> list[str]:
    L = [f"REGISTERED {d['registered']}: rental 3, {d['name']}", f"status: {d['status']}", f"design: {d['design']}",
         f"tool: {d['tool']}", ""]
    if part == "e2e":
        L.append(f"scope: {d['scope']}")
        L += ["", "predicted T (ms), M and M+Z, the core population (host rule passed):"]
        for c in d["population"]["core"]:
            L.append(f"  {c['arm']:8s} G={c['G']:<3} n={c['n']}  M {c['T_M']:.4f}  M+Z {c['T_MZ']:.4f}")
        L.append(f"  core {d['population']['core_cells']} cells, deep {d['population']['deep_cells']} cells")
        L += ["", "predicted PRIVATE / SHARED q (w1 / w2), byte G's:"]
        for k, v in d["q_pred"].items():
            L.append(f"  {k:20s} {v['w1'] if v['w1'] is None else round(v['w1'], 4)} / {v['w2'] if v['w2'] is None else round(v['w2'], 4)}")
    if part == "floorlaw":
        L.append("registered floor-bound cells:")
        for m, cs in d["cells_registered"].items():
            for g in ("w1", "w2"):
                L.append(f"  {m} {g}: n " + ", ".join(str(c["n"]) for c in cs if c["gemm"] == g and c["floor_bound"]))
        L += ["", "SEEN rulers (rental 2), theta +- se per ruler:"]
        for e in d["seen_rulers"]:
            L.append(f"  {e['capture']:32s} {e['gemm']} " + "  ".join(
                f"{k} {e[k]['theta']:+.2f}+-{e[k]['se']:.2f}" for k in ("ACT.max", "DUR", "EL.avg", "X", "GAP")))
    if part == "zform":
        for k, v in d["fits_CAL"].items():
            L.append(f"  CAL {k:5s} coef {v['coef']}  chi2/dof {v['chi2_dof']}")
        L.append(f"  SEEN residual rms (cycles): {d['seen_residuals']['rms']} over {d['seen_residuals']['series']} series")
    L.append("")
    for k, v in d.items():
        if k in SKIP_TXT or k in ("registered", "name", "status", "design", "tool", "scope", "population"):
            continue
        _flat("", {k: v}, L)
    return L


TXT_PARTS = C3.PARTS


def build() -> dict:
    cx = Ctx()
    series = published_series(cx)
    fits = z_fits(series)
    docs = {"e2e": e2e(cx), "floorlaw": floorlaw(cx), "zform": zform(cx, series, fits),
            "stages": stages(), "flush": flush(), "replicate": replicate(cx)}
    pins = {p: sha256(REPO / p) for p in (
        "scripts/cross_model_score.py", "scripts/cross_model_predict.py", "scripts/r3_timing_model.py",
        "scripts/wave_split_bytes.py", "scripts/floor_estimator.py", "scripts/l2_survival.py",
        "scripts/scoring/rental3/r3common.py")}
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
        jpath = a.out_dir / f"{C3.NAMES[part]}.json"
        tpath = a.out_dir / f"{C3.NAMES[part]}.txt"
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
