"""Rental 6's four registrations (docs/registered/2026-10-09-rental6-{v3,skew,q1,secondk}-gh200),
built for scripts/scoring/rental6/register.py, which renders, pins and --checks them.

Every number is computed here from committed files (the timing model with v3's flag, the byte
model with MODEL M4, the published pages, the synthetic histogram pages and the row-key rival's
file) or is a typed-in input with its provenance and its label (CAL, CAL-counters, SEEN,
SEEN-FITTED, SEEN-SELECTED, SEEN-INFORMED, IN-SAMPLE, BLIND, BLIND-CALMODEL).

Design: scratchpad design-r6/DESIGN.md, corrected by design-r6-review/REVIEW.md (the review wins
where they conflict), with the owner's decisions of 2026-10-09: 1 (b) only where NATIVE takes
block-scan; 2 h from the four-model S - N gaps, kept and labelled SEEN-FITTED; 3 E1-SKEW at the
time bar; 4 Phi carries Mixtral's synthetic shapes; 5 a cell needs 6 of 9 clean repeats and a page
at most 10% slipped repeats; 6 no permutation cells. 8x22B, Qwen2-57B and OLMoE are IN-SAMPLE for
(b); content_a is SEEN-SELECTED; the n = 1 exclusion is SEEN-INFORMED.

LEAKAGE, said once for the four: every published page is SEEN; every constant typed in below was
fitted or chosen after seeing published pages (labelled per entry). No rental-6 page exists when
this runs.
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r6common as C6  # noqa: E402
import r6model as M6  # noqa: E402
import r3_timing_model_v3 as TV3  # noqa: E402

C4, C5 = C6.C4, C6.C5
REPO = C6.REPO
HIST = REPO / C6.HIST_DIR
r6 = C6.r6
DESIGN = ("design-r6 DESIGN.md corrected by design-r6-review REVIEW.md (a)-(h) (the review wins) and the owner's "
          "decisions of 2026-10-09 (1-6 accepted; h kept and labelled SEEN-FITTED)")
REGISTERED = f"{C6.DATE}, before any rental-6 page"

SHAPES = {"mixtral-8x7b": (3, 6, 16), "olmoe-1b-7b": (3, 6, 12), "phi-3.5-moe": (2, 4, 12)}
TIMED_PAGES = ("A", "B", "C")
BYTE_MODELS = ("olmoe-1b-7b", "phi-3.5-moe")
SKEWED = ("PT", "PW", "PW2", "DW")
Q1 = {"phi-3.5-moe": (1, 2), "qwen1.5-moe-a2.7b": (1, 2), "jetmoe-8b": (1, 2, 4)}
COPIES = (9, 15)
RIVALS_E1 = ("v2", "LT3", "H8", "v3_all", "v3_noca")
RIVALS_GAP = ("v2", "v3_all", "H8")
#: replicate sigma of one pair's g, us (design-r6-review reps.json: the rental-5 page A - B
#: replicate; SEEN). Phi and JetMoE have no replicate: 1.0 and 0.45 are the review's assumption.
SIG_US = {"olmoe-1b-7b": 0.20, "qwen1.5-moe-a2.7b": 0.45, "mixtral-8x7b": 2.04, "phi-3.5-moe": 1.0, "jetmoe-8b": 0.45}
#: SEEN v3 gap residual g at G 8 on balanced cells (design-r6-review mc_gap.py SEEN, from refit_h)
SEEN_ANCHOR_US = {("phi-3.5-moe", 2): -0.74, ("qwen1.5-moe-a2.7b", 2): 0.90, ("olmoe-1b-7b", 3): -1.54,
                  ("mixtral-8x7b", 3): -8.03, ("phi-3.5-moe", 1): 0.32}
TAU_US = 1.3
#: Q1-H noise sensitivity (build review section 3 fix 2): (Phi sd, JetMoE sd) us
Q1H_SENS = ((1.0, 0.45), (2.04, 0.45), (2.04, 2.04))
#: the same three points re-run on the laptop GPU (torch MPS, float32) by the build, against these
#: CPU float64 numbers (typed in: register.py runs on the CPU only)
Q1H_SENS_MPS = {"device": "mps", "dtype": "float32", "torch": "2.13.0", "R": 200_000, "seed": 20261010,
                "P(v3 HOLDS)": {"phi 1.0 us, jetmoe 0.45 us": 0.9926, "phi 2.04 us, jetmoe 0.45 us": 0.864,
                                "phi 2.04 us, jetmoe 2.04 us": 0.7237},
                "max_abs_diff_vs_cpu_float64": 0.0001, "seconds_mps": 0.68, "seconds_cpu": 0.14}
#: rental-5 skewed-minus-control gap offsets (design-r6-review reps.json; SEEN)
SKEW_OFFSET_SEEN = {"olmoe-1b-7b": {"mean_us": -3.39, "rms_us": 4.41}, "qwen1.5-moe-a2.7b": {"mean_us": -1.78, "rms_us": 2.40},
                    "mixtral-8x7b": {"mean_us": -0.41, "rms_us": 4.98}}
#: h's estimate (design-r6 work/hrule.py, h_cal.json): SEEN-FITTED
H_FIT = {"h_us": -5.16, "h_unrounded_us": -5.156793587769217, "page_bootstrap_95_us": [-6.21, -4.30],
         "s_hat_us": -1.912, "s_hat_ci_us": [-2.773, -0.864], "cells": 178, "groups": 27,
         "per_model_s_hat_us": {"mixtral-8x22b": 0.12, "mixtral-8x7b": -1.37, "olmoe-1b-7b": -1.43, "qwen2-57b-a14b": -3.86},
         "estimator": "per block-scan cell s_hat = (S - N)_meas - (S - N)_v2 + A + s_block_v2 under rental 3's host rule "
                      "(T_N + F240 >= 0.40 ms); h = -(mean s_hat - s_block_v2); page-cluster bootstrap B 20000 seed 20261009"}
H_GROUPS = [
    {"pages": "8x7B 2026-09-27 G 2/3/4/8/32", "cells": 35, "status": "CAL for the 5 registered pages; G2 and G4 repeat pages SEEN"},
    {"pages": "8x7B 2026-09-25 G 1/2/4/16/64", "cells": 15, "status": "SEEN, not in the CAL timing set"},
    {"pages": "8x22B 2026-09-28", "cells": 26, "status": "SEEN held-out TIME test page"},
    {"pages": "Qwen2-57B 2026-09-28", "cells": 53, "status": "SEEN held-out TIME test page"},
    {"pages": "OLMoE 2026-09-29", "cells": 33, "status": "SEEN held-out TIME test page (rental5-v2: no OLMoE time page enters any v2 fit)"},
    {"pages": "OLMoE rental 4 (2026-10-07) D 576 and 960", "cells": 16, "status": "SEEN held-out TIME test page"},
]


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha_counts(counts) -> str:
    return hashlib.sha256(json.dumps(None if counts is None else [int(v) for v in counts],
                                     separators=(",", ":")).encode()).hexdigest()


def load(name: str) -> dict:
    return json.loads((HIST / f"{name}.json").read_text())


def _arms(c: dict) -> list:
    return list(c.get("arms") or ("native", "shared"))


# --------------------------------------------------------------------------
# pricing (once per process)
# --------------------------------------------------------------------------

_P: list = []


def pricer() -> M6.Pricer:
    if not _P:
        _P.append(M6.Pricer())
    return _P[0]


_CACHE: dict = {}


def priced(model, arm, n, counts, copies, seed) -> dict:
    key = (model, arm, n, None if counts is None else tuple(counts), copies, seed)
    if key not in _CACHE:
        _CACHE[key] = pricer().price(model, arm, n, counts, copies, seed)
    return _CACHE[key]


def _ms(p: dict) -> dict:
    return {v: r6(p[v]) for v in M6.VARIANTS}


def label_timed(model: str, label: str, n: int, copies: int = 9, arm: str = "shared") -> str:
    if model == "mixtral-8x7b":
        if label == "balanced" and n in (3, 6):
            return "SEEN replicate (2026-09-27 G 8 CAL pages carry this balanced cell; only the board is new)"
        return "BLIND-CALMODEL (8x7B is the CAL timing model; no cell at n 3, 6, 16 with a histogram exists)"
    if model == "olmoe-1b-7b":
        if label == "balanced" and n <= 9:
            return "SEEN replicate (2026-09-29 and rental-4 G 8 pages); IN-SAMPLE for (b)"
        return "BLIND draw, IN-SAMPLE for (b) (h is fitted on OLMoE's 2026-09-29 and rental-4 timed pages)"
    if model == "phi-3.5-moe":
        if copies == 15 and arm == "native":
            return c15_native_seen(model)
        if copies == 9 and label == "balanced" and n in (1, 2, 4):
            return "SEEN replicate (2026-09-29 c9 G 8 balanced pages, probe on); only the board is new"
        return "BLIND"
    if model in ("jetmoe-8b", "qwen1.5-moe-a2.7b"):
        if copies == 15:
            return c15_native_seen(model) if arm == "native" else "BLIND (c15 SHARED: no c15 page exists)"
        return "SEEN replicate (2026-09-29 / rental-5 c9 pages)"
    return "BLIND"


#: NATIVE declares E whatever the copies count, so a c15 page's NATIVE cell is the c9 page's
#: NATIVE cell (design-r6-review build review, section 3 fix 1: the 2026-09-29 Phi, Qwen1.5 and
#: JetMoE pages, and rental 5 for Qwen1.5, carry it). Only c15 SHARED is blind.
def c15_native_seen(model: str) -> str:
    also = " and rental 5's" if model == "qwen1.5-moe-a2.7b" else ""
    return (f"SEEN replicate (c15 NATIVE declares E at any copies count, so it repeats the 2026-09-29{also} c9 NATIVE "
            "cell); only the board is new")


def timed_cells() -> list[dict]:
    """Every planned timed cell: the nine skew pages (E1) and the six Q1 pages."""
    rows = []
    for model, ns in SHAPES.items():
        for page in TIMED_PAGES:
            doc = load(f"{model}-{page}")
            seed = doc["shuffle_seed"]
            for c in doc["cells"]:
                for arm in _arms(c):
                    p = priced(model, arm, c["n"], c["counts"], 9, None if c["counts"] is None else seed)
                    rows.append({"test": "E1", "model": model, "page": page, "copies": 9, "n": c["n"], "label": c["label"],
                                 "arm": arm, "counts_sha256": sha_counts(c["counts"]), "ms": _ms(p),
                                 "A_proxy_us": p["A_us"], "A_how": p["A_how"], "path": p["path"],
                                 "native_path": p["native_path"], "declared": p["declared"], "dead_ctas": p["dead"],
                                 "host_ok": C6.host_ok(p["v3"]), "label_blind": label_timed(model, c["label"], c["n"])})
    for model, ns in Q1.items():
        doc = load(f"q1-{model}")
        for copies in COPIES:
            for c in doc["cells"]:
                for arm in _arms(c):
                    p = priced(model, arm, c["n"], None, copies, None)
                    rows.append({"test": "Q1", "model": model, "page": f"c{copies}", "copies": copies, "n": c["n"],
                                 "label": "balanced", "arm": arm, "counts_sha256": sha_counts(None), "ms": _ms(p),
                                 "A_proxy_us": p["A_us"], "A_how": p["A_how"], "path": p["path"],
                                 "native_path": p["native_path"], "declared": p["declared"], "dead_ctas": p["dead"],
                                 "host_ok": C6.host_ok(p["v3"]), "label_blind": label_timed(model, "balanced", c["n"], copies, arm)})
    return rows


def _key(r) -> tuple:
    return (r["model"], r["page"], r["n"], r["label"], r["arm"])


# --------------------------------------------------------------------------
# rental6-v3: the model (no unit)
# --------------------------------------------------------------------------

def seen_reproduction() -> dict:
    """v3 and every rival on the rental-5 SEEN skew pages (72 cells, A from the proxy) and v3's
    in-sample read of the 8x7B CAL fitted cells: the design's numbers, through the repo's flags."""
    import r3_timing_model as TM
    P = pricer()
    hist = {}
    for f in sorted(glob.glob(str(REPO / "docs/registered/2026-10-07-rental5-skew-hist/*-[AB].json"))):
        d = json.loads(Path(f).read_text())
        for c in d["cells"]:
            hist[(Path(f).stem, c["label"], int(c["n"]))] = (c.get("counts"), d.get("shuffle_seed"))
    R5 = REPO / "results/published/2026-10-09-nvidia_gh200_480gb-rental5-session/results"
    res = []
    for m in ("olmoe-1b-7b", "qwen1.5-moe-a2.7b"):
        for pg in "AB":
            rp = sorted(glob.glob(str(R5 / f"gaps-nvidia_gh200_480gb-{m}-sk{pg.lower()}/private_weight_reference/*/report.json")))[0]
            for t in json.loads(Path(rp).read_text())["treads_table"]:
                key = (f"{m}-{pg}", t["histogram"], int(t["tiles"]))
                if key not in hist or t["histogram"] == "balanced":
                    continue
                counts, seed = hist[key]
                p = P.price(m, t["arm"], int(t["tiles"]), counts, 9, seed)
                if p is not None:
                    res.append({v: p[v] / t["ms_p50"] - 1 for v in M6.VARIANTS})
    out = {"rental5_E1_SKEW_72": {v: C5.stats([r[v] for r in res]) for v in M6.VARIANTS}, "cells": len(res)}
    al = P.cal_align()
    cells = [c for c in P.eng.src2["cells"] if c.fit]
    e = []
    old = TM.set_model("mixtral-8x7b")
    try:
        with TM.dead_model("v2"):
            for c in cells:
                A = None if c.arm == "native" else st.median(al[r][(c.arm, c.n)] - al[r][("native", c.n)] for r in c.runs)
                e.append(TV3.call_ms(P.f2.x, c, P.ctx, P.f2.k_w, align_ms=A,
                                     native_path=P.eng.path("mixtral-8x7b", "native", c.n, 9)) / c.ms - 1)
    finally:
        TM.set_model(old)
    out["cal_8x7b_fitted_cells_v3"] = C5.stats(e)
    out["design_numbers"] = {"rental5_E1_SKEW_72_v2": "1.26 / +0.40 / +4.38", "rental5_E1_SKEW_72_v3": "0.90 / +0.09 / +3.08",
                             "rental5_E1_SKEW_72_H8": 1.40, "cal_8x7b_v3_rms": 0.249}
    return out


def v3() -> dict:
    P = pricer()
    proxy = P.proxy
    return {
        "registered": REGISTERED, "name": C6.NAMES["v3"], "status": "REGISTERED BEFORE ANY PAGE", "design": DESIGN,
        "tool": "scripts/r3_timing_model_v3.py (V3_H_MS, v3_b_ms, call_ms), scripts/wave_split_bytes_v3.py (ModelV3, "
                "content_events_counts: MODEL M4), scripts/scoring/rental6/r6model.py (v3 and the rivals)",
        "modules": {"rule": "OWNER DECISION 2026-10-09: v3 lives in new modules that import the pinned ones; "
                            "scripts/r3_timing_model.py, scripts/wave_split_bytes.py and scripts/scoring/rental5/skewmodel.py "
                            "stay byte-identical to rental5-skew's code_pins (tests/test_rental6_model.py checks the sha256)",
                    "timing": "r3_timing_model_v3.call_ms = r3_timing_model.call_ms (dead_model v2) + term (b)",
                    "bytes": "wave_split_bytes_v3.ModelV3 = wave_split_bytes.Model(content_a=True) + content keys on histogram cells"},
        "model": {
            "base": "model v2 (docs/registered/2026-10-07-rental5-v2-gh200), its registered x unchanged: " + json.dumps(
                {k: r6(v) for k, v in P.f2.params.items()} if isinstance(P.f2.params, dict) else str(P.f2.params)),
            "a_content_a": "MODEL M4: w1's A keyed by tile CONTENT on histogram cells too (realize_counts, the page's "
                           "token-row shuffle, each expert's rows sorted; partial last tiles weighted rows / 32; the "
                           "compulsory offset (distinct-content rows - all tile rows) / 32 A tiles); 0 fitted constants. "
                           "On a skewed cell no two experts share a token set, so it moves w1 q only through the partial-"
                           "tile weighting (at most 0.16% on the rental-6 byte cells); it moves uniform / balanced w1 "
                           "(OLMoE up to 16%, Mixtral at most 0.29%, Phi at most 0.71%)",
            "b_alignment": "on every non-NATIVE cell whose NATIVE twin (same model, n, declaration E) takes the "
                           "block-scan alignment kernel: + A_arm(n) from the page's own align_probe, and on SHARED "
                           f"also + h = {C6.r6(1e3 * TV3.V3_H_MS)} us; 0 elsewhere",
            "A_statistic": "A_arm(n) = median over the probe's repeats of align(arm, n) ms - median of align(native, n) "
                           "ms, both from the same page's align_probe; histogram-independent (the probe runs balanced "
                           "ids at the tread's numel, and a histogram cell's numel is its tread's), so a histogram "
                           "cell takes A at its tread n",
        },
        "h": {"value_us": H_FIT["h_us"], "label": "SEEN-FITTED (4 models' timed pages): at least 143 of its 178 cells are "
                                                  "SEEN held-out timing pages of earlier registrations",
              "fit": H_FIT, "groups": H_GROUPS,
              "in_sample": "8x22B, Qwen2-57B and OLMoE (and 8x7B) are IN-SAMPLE for term (b); OLMoE's E1 cells and its "
                           "n 3 Q1-H pairs are therefore not BLIND-CALMODEL for (b)",
              "rival_H8": {"what": "the registered 8x7B timing set alone, h free (6 parameters), A from those pages' probes",
                           "x": {k: r6(v) for k, v in zip(("T0_ms", "c_ns", "bw_gbps", "s_small_ms", "s_block_ms"), P.h8["x"], strict=True)},
                           "h_us": r6(1e3 * P.h8["h_ms"]), "rms": r6(P.h8["rms"]), "cells": P.h8["cells"]}},
        "seen_informed_choices": [
            "h = -5.16 us is SEEN-FITTED (above); kept by the owner (2026-10-09)",
            "the block-scan path condition is SEEN-INFORMED: chosen after the rental-5 Mixtral n 2 pages (slip-affected) read "
            "+1.6 us under v3 and +9.9 us with (b) everywhere. Evidence the other way, stated: on the CAL 2026-09-27 G 8 page "
            "Mixtral n 3 (NATIVE small-batch) reads g = -8.03 us under v3, the v3_all direction",
            "content_a is SEEN-SELECTED: picked out of 14 diag-Q2 WSC variants on SEEN byte pages (rental 5 skc, rental 4 B', "
            "rental 1 A-tile)",
            "the n = 1 exclusion from Q1-H is SEEN-INFORMED: Qwen1.5 SHARED n 1 was seen to be anomalous (+6.6 us) on the "
            "2026-09-29 and rental-5 pages, and CAL Qwen2-57B reads the same at n 1",
            "Phi-3.5 carries Mixtral's fitted (kind, param) to E 16 at Phi's B (owner decision 4): no Phi routing statistics",
            "the Q1-H population is control pairs only (review (b)): skewed pairs carry a SEEN offset (Q1-SK, open)",
            "the Q1-P statistic is DESCRIPTIVE (rental6-q1): its SEEN small-batch residual trends with n",
            "B-SKEW carves OLMoE's skewed w1 rows out of its pooled verdict (OPEN-L2SKEW, scored apart): SEEN-INFORMED, "
            "decided after rental 5 read those rows +8% at n 4 and +25% at n 16 (v3's content keys do not move a skewed cell)",
        ],
        "rental5_code_pins_note": ("rental 5's registrations list scripts/private_weight_reference.py and scripts/skew_synth.py "
                                   "in their code_pins, and rental 6 changes both (the slip plumbing and the align probe; "
                                   "--draw6 / --check6). Harmless: rental 5's scorer (scripts/scoring/rental5/score_skew.py, "
                                   "unchanged) checks only its three pinned files, r3_timing_model.py, wave_split_bytes.py "
                                   "and rental5/skewmodel.py, which stay byte-identical (tests/test_rental6_model.py), and "
                                   "rental 5's pages are already published; the other pins record the tree, they gate nothing"),
        "align_proxy": {"what": "the SEEN-table proxy (review (h) 17): A when a page has no align_probe, and the A every "
                                "registered prediction carries; the median over every published GH200 BLOCK_M 32 report's "
                                "probe at that (model, SHARED declaration); n > 9 extrapolated along n 8 -> 9; a declaration "
                                "with no table takes the c9 table", "table": proxy},
        "seen_reproduction": seen_reproduction(),
        "open_residuals": {
            "Q1-C": "Qwen1.5 SHARED n = 1 carries a further ~+6.6 us under v3 (SEEN); CAL Qwen2-57B reads the same at n 1; "
                    "registered as open with rivals CALL and DEAD on D = x15 - x9 (rental6-q1)",
            "Q1-SK": "skewed pairs' S - N gap sits under the controls' on the same page (SEEN: OLMoE -3.39 us, Qwen1.5 -1.78 "
                     "us, Mixtral rms 4.98 us); open, printed (rental6-q1)",
            "OPEN-L2SKEW": "skewed OLMoE w1 bytes: v3's content keys do not move a skewed cell; the row-key rival is priced "
                           "before any page (rental6-skew)",
            "granite": "the 09-30 Granite G 8 cells with n <= 6 are host-bound (268-303 us every arm): nothing planned",
        },
        "variants": {"v2": "registered model", "v3_nob": "v3 without (b)", "v3_noca": "v3 without (a)",
                     "LT3": "v3 with wave_q = 1", "v3_all": "v3 with (b) on every non-NATIVE cell",
                     "H8": "M4 bytes, the 8x7B 6-parameter refit's (x, h), (b) on every non-NATIVE cell"},
        "seen_data": ["the v2 source fit is CAL (8x7B 2026-09-27)", "h is SEEN-FITTED", "content_a SEEN-SELECTED",
                      "the align proxy is SEEN", "every reproduction number is on SEEN pages"],
    }


# --------------------------------------------------------------------------
# rental6-skew
# --------------------------------------------------------------------------

#: OWNER DECISION 2026-10-09 on a G3 failure, and the build review's risk estimate
G3_VOID_ALL = {
    "rule": "OWNER DECISION 2026-10-09: if G3 FAILS and Cochran Q does not reject, every histogram cell is voided and "
            "E1-SKEW reads NOT SCORED (rental 5's strict rule kept). E1 is ALSO printed DESCRIPTIVELY on the voided cells "
            "(score_skew.py: E1-SKEW 'descriptive', the same statistic and the same bars, labelled 'DESCRIPTIVE, NOT A "
            "VERDICT'); it decides nothing and is not the registered E1-SKEW",
    "risk": "the design stated P(G3 passes with Q not rejecting) 0.94, a 6% risk. The build review's simulation (OLMoE at "
            "its SEEN rental-5 token-order effect -0.24%, Mixtral and Phi at the N(-0.13%, 0.15%) prior, cell noise sqrt2 x "
            "0.18%, cluster = page, df 8) gives 0.79, and 0.65 to 0.70 under harsher assumptions (OLMoE at -0.30%, or a "
            "0.1% page sd): about 21 to 35% that G3 does not pass cleanly, mostly from OLMoE's -0.24%. Of that, the "
            "void-everything branch (the TOST fails, Q does not reject) is about 9 to 14%; the rest is Q rejecting, where "
            "the per-stratum TOSTs void only the failing strata",
    "provenance": "build-r6-review g3_sim.py (seed 1, R 100000): registered prior P(pass, Q ok) 0.791, P(Q rejects) 0.122; "
                  "OLMoE -0.30%: 0.699, 0.166; + page sd 0.1%: 0.646, 0.231",
}

def _files(names) -> dict:
    out = {}
    for nm in names:
        p = HIST / f"{nm}.json"
        d = json.loads(p.read_text())
        out[f"{nm}.json"] = {"path": str(p.relative_to(REPO)), "sha256": sha256_file(p), "shuffle_seed": d["shuffle_seed"],
                             "cells": [{"label": c["label"], "n": c["n"], "arms": _arms(c), "counts_sha256": sha_counts(c["counts"]),
                                        **({"seed": c["provenance"]["seed"], "kind": c["provenance"]["kind"],
                                            "param": c["provenance"]["param"]} if c.get("provenance") else {})}
                                       for c in d["cells"]]}
    return out


def _levers(rows: list[dict]) -> dict:
    thr = C6.LEVER_K * C6.SIGMA_PAGE
    out = {}
    for R in RIVALS_E1:
        out[R] = sorted("/".join(map(str, _key(r))) for r in rows
                        if r["test"] == "E1" and r["label"] != "balanced" and abs(r["ms"][R] / r["ms"]["v3"] - 1) > thr)
    return out


def byte_predictions() -> list[dict]:
    import r3_timing_model as TM
    rk = json.loads((HIST / "rowkey_rival.json").read_text())
    rkc = {(c["label"], c["n"], c["arm"]): c for c in rk["cells"]}
    out = []
    for model in BYTE_MODELS:
        doc = load(f"{model}-bytes")
        old = TM.set_model(model)
        bm = dict(TM.BYTE_MODEL)
        TM.set_model(old)
        for c in doc["cells"]:
            for arm in _arms(c):
                p = priced(model, arm, c["n"], c["counts"], 9, None if c["counts"] is None else doc["shuffle_seed"])
                q = {v: {g: r6((p[f"bytes_{v}"][g] - c["n"] * bm[f"operand_per_tile_{g}"]) / bm[f"W_{g}"]) for g in ("w1", "w2")}
                     for v in ("v2", "v3")}
                row = {"model": model, "n": c["n"], "label": c["label"], "arm": arm, "counts_sha256": sha_counts(c["counts"]),
                       "q_pred": q["v3"], "q_pred_v3_noca": q["v2"], "read_bytes_v3": {g: r6(x) for g, x in p["bytes_v3"].items()},
                       "label_blind": ("SEEN replicate" if c["label"] == "balanced" and (model, c["n"]) in (("olmoe-1b-7b", 3), ("olmoe-1b-7b", 6), ("phi-3.5-moe", 2), ("phi-3.5-moe", 4))
                                       else "BLIND draw" + ("; OLMoE is CAL-counters (d, kappa)" if model == "olmoe-1b-7b" else ""))}
                if model == "olmoe-1b-7b" and (c["label"], c["n"], arm) in rkc:
                    rr = rkc[(c["label"], c["n"], arm)]
                    row["rowkey_w1_read_bytes"] = {f"C{C:g}": rr[f"read_bytes_C{C:g}"] for C in rk["C_MiB"]}
                    row["rowkey_w1_q"] = {f"C{C:g}": r6((rr[f"read_bytes_C{C:g}"] - c["n"] * bm["operand_per_tile_w1"]) / bm["W_w1"]) for C in rk["C_MiB"]}
                    row["rowkey_over_v3_w1"] = r6(rr[f"read_bytes_C{rk['registered_C_MiB']:g}"] / p["bytes_v3"]["w1"] - 1)
                out.append(row)
    return out


def skew() -> dict:
    rows = timed_cells()
    E1 = [r for r in rows if r["test"] == "E1"]
    k = {_key(r): r for r in E1}
    for r in E1:
        u = k.get((r["model"], r["page"], r["n"], "uniform", r["arm"]))
        if r["label"] in SKEWED and u is not None:
            r["ratio"] = {v: r6(r["ms"][v] / u["ms"][v]) for v in M6.VARIANTS}
            r["lever_ratio"] = {"LT3": abs(r["ratio"]["LT3"] - r["ratio"]["v3"]) > 0.0075, "U": abs(r["ratio"]["v3"] - 1) > 0.045}
        b = k.get((r["model"], r["page"], r["n"], "balanced", r["arm"]))
        if r["label"] == "uniform" and b is not None:
            r["delta_v3"] = r6(r["ms"]["v3"] / b["ms"]["v3"] - 1)
    lev = _levers(E1)
    for r in E1:
        r["lever"] = {R: "/".join(map(str, _key(r))) in lev[R] for R in RIVALS_E1}
    rk = json.loads((HIST / "rowkey_rival.json").read_text())
    summ = {}
    for m in SHAPES:
        rr = [r["ratio"]["v3"] - 1 for r in E1 if r["model"] == m and "ratio" in r]
        summ[m] = {"r_v3_minus_1_pct": [r6(100 * min(rr)), r6(100 * max(rr))],
                   "T_v3_us": [r6(1e3 * min(r["ms"]["v3"] for r in E1 if r["model"] == m)), r6(1e3 * max(r["ms"]["v3"] for r in E1 if r["model"] == m))]}
    gaps = {}
    for R in RIVALS_E1 + ("v3_nob",):
        g = [r["ms"][R] / r["ms"]["v3"] - 1 for r in E1 if r["label"] != "balanced"]
        gaps[R] = {"cells": len(g), "rms_pct": r6(100 * math.sqrt(st.fmean(x * x for x in g))), "worst_pct": r6(100 * max(g, key=abs)),
                   "lever_cells_0.18": len(lev.get(R, [])) if R in lev else None}
    return {
        "registered": REGISTERED, "name": C6.NAMES["skew"], "status": "REGISTERED BEFORE ANY PAGE", "design": DESIGN,
        "tool": "scorer scripts/scoring/rental6/score_skew.py; predictions r6model.py via reg6.py; histograms scripts/skew_synth.py --draw6",
        "model": "model v3 (docs/registered/2026-10-09-rental6-v3-gh200); every cell carries v2, v3, v3_nob, v3_noca, LT3, v3_all and H8 (ms)",
        "histograms": {"synthetic": "SYNTHETIC: rental 5's fitted Zipf / Dirichlet shapes, (kind, param) log-B interpolated to the new treads "
                                    "(Phi: Mixtral's shapes at Phi's B, owner decision 4), one deterministic draw per (model, n, label) at seed "
                                    "20261010 + 1000 x shape index + 10 n + label index (PT 0, PW 1, DW 2, PW2 3); shuffle seed 20261010",
                       "files": _files([f"{m}-{p}" for m in SHAPES for p in TIMED_PAGES] + [f"{m}-bytes" for m in BYTE_MODELS])},
        "cells": {"common": "R3's tile 32x64x64 w8 s4 at G = 8; NATIVE and SHARED at 9 copies (not vLLM's tile: nothing here is the working bar on vLLM's tile)",
                  "pages": {"A": "PT, PW, uniform, balanced", "B": "DW, uniform, balanced", "C": "PW2 (a second PW draw), uniform, balanced"},
                  "treads": {m: list(v) for m, v in SHAPES.items()},
                  "n_stratum": "G3's stratum is the tread's rank within its shape",
                  "dropped": "the permutation cells (owner decision 6); Mixtral's byte page (review (c): lever at most 0.29%)"},
        "predictions": E1,
        "byte_predictions": byte_predictions(),
        "rowkey_rival": {"file": str((HIST / "rowkey_rival.json").relative_to(REPO)), "sha256": sha256_file(HIST / "rowkey_rival.json"),
                         "tool": "scripts/scoring/rental6/rowkeys.py", "device": rk["device"], "dtype": rk["dtype"],
                         "cpu_float64_check": rk["cpu_float64_check"], "registered_C_MiB": rk["registered_C_MiB"]},
        "summary": summ, "rival_gaps_on_E1": gaps, "levers": {"rule": f"|T_R / T_v3 - 1| > 3 sigma_page = {C6.LEVER_K * C6.SIGMA_PAGE:.2%} (sigma_page 0.18%, rental 3 rep8)", "cells": lev},
        "gates": {
            "G1_lock_thermal": "R3's G1_lock_thermal PASS and the page's slipped repeat rows (R3 lock_slip_repeats) at most 10% of its repeat rows; "
                               "a HELD_WITH_SLIPS page within that passes; a page failing it counts in ALL and is dropped in CLEAN (rental 2's rule 3)",
            "repeats": "a cell is VALID with at least 6 clean repeats of 9 (clean: not lock-slipped, not drift-excluded); its time and clock are R3's "
                       "medians over the clean repeats; else the cell is dropped; every slipped repeat is printed with its clock",
            "G2_host_bound": "rental 3's registered host rule (T_v3 + F240 >= 0.40 ms on the REGISTERED prediction, F240 0.067551 ms; "
                             "every rental-6 E1 cell passes it) AND the page's timer flag host_bound; a pair needs both cells",
            "G4_cell_clock": "the cell's clean-repeat median clock at least 1705 MHz, else the CELL is dropped (a cell in [1695, 1705) too)",
            "G5_provenance": "as rental 5: histogram file sha256, every row's counts_sha256, bincount_ok, the shuffle seed; else the page is NOT SCORED",
            "G3": "rental 5's text verbatim, with delta_v3 (the registered uniform / balanced - 1 under v3, 0 on every pair): d_i = "
                  "T(uniform)/T(balanced) - 1 - delta_v3,i on every valid (shape, page, n, arm), NATIVE and SHARED pooled; equivalence "
                  "if the 90% cluster-t CI of mean(d), clustered on page (df = pages - 1), lies within +-0.36%; Q = sum (d_i - mean)^2 / "
                  "(2 x 0.0018^2) against chi2(N - 1); if p < 0.01 the TOST per n-stratum voids only the failing strata's histogram "
                  "cells, otherwise a failure voids every histogram cell; unseen shapes' token-order prior N(-0.13%, 0.15%)",
            "G3_void_all": G3_VOID_ALL,
            "no_report": "a page with no report.json is NOT SCORED; cells.csv is never read",
            "views": "ALL and CLEAN (rental 2's rule 3, r3common.two_views); the registered verdict is the combination rule's",
        },
        "counter_clock_rule": {
            "rule": "lock_gate.check_page PASS (nvidia-smi before and after within 15 MHz of the lock, the null kernel and base twin "
                    "where present), then each cell's SM clock (sm__cycles_elapsed.avg / gpu_time_ns, median over its GEMMs of at "
                    "least 0.5 ms, else the page's median) in [1640, 1710] MHz; a cell outside is dropped and the page continues",
            "replaces": "the unmeetable 1705 MHz cut on counter pages (cycles over duration is a lower bound on the clock)",
            "derivation": "the smi-held lock-1710 published counter pages: 3025 of 3033 cells (99.74%) pass; the one off-lock page "
                          "(the 8x22B floor, smi 1980 MHz) fails check_page and 0 of its 4 cells are in the band",
            "under_lock_control": "the published floor pages locked UNDER 1710 are the under-lock negative control: the six "
                                  "floor1005 pages (rental 2 and rental 3: Mixtral tp2 / tp4 / tp8, Qwen2-57B tp8, Granite) and "
                                  "the rental-3 Mixtral tp4 floor1410 page each PASS check_page (their own lock is in force) "
                                  "and give 0 cells in [1640, 1710] MHz (cells at 1004-1005 and 1396-1405 MHz; Granite's "
                                  "cells carry no clock); tests/test_scoring_rental6.py checks it",
            "limits": "the under-lock control pages sit 18% and 41% under the lock: the band cannot see a sag under about 4%; it "
                      "was derived from smi-held pages only, and the smi bracket cannot see a sag inside a capture",
        },
        "tests": {
            "E1-SKEW": {"statistic": "e = T_v3 / T_meas - 1 (T_v3 the registered prediction carried to the page's own A: "
                                     "r6model.reprice) over every valid histogram cell (PT PW PW2 DW uniform)",
                        "rule": "HOLDS if rms <= 2%, max |e| <= 5% (the time bar) and |mean e| <= 1% (owner decision 3), each over the "
                                "POOLED valid cells (all shapes, both arms); per shape and per arm printed; in ALL and CLEAN",
                        "on_G3_void_all": "NOT SCORED; E1 is printed DESCRIPTIVELY on the voided cells, not a verdict "
                                          "(gates.G3_void_all, owner decision 2026-10-09)",
                        **C6.E1_BAR, "rental5_bands_printed": C6.E1_R5_BANDS,
                        "power": "a bar test, not a discriminating test (design 3.2: P(holds) 1.0 under v3 and under v2, H8, v3_all; LT3 0.995)"},
            "E1-UNI": {"statistic": "e on the balanced cells", "rule": "the E1-SKEW bar, printed"},
            "E1-rivals": {"rule": "rental 5's: a rival is EXCLUDED if its rms on its lever cells is at least 2x v3's there and at least 0.76%; "
                                  "UNDECIDED below 6 surviving lever cells", "rivals": list(RIVALS_E1),
                          "rival_factor": C6.RIVAL_FACTOR, "rival_floor": C6.RIVAL_FLOOR, "min_lever_cells": C6.MIN_LEVER},
            "SKEW-RATIO": {"statistic": "r = T(h) / T(uniform), same page, n and arm; e = r_v3 / r_meas - 1 (A cancels on a block-scan pair)",
                           "rule": "rental 5's: HOLDS if rms <= 1.5% and max |e| <= 4%; LT3 and U by the rival rule on their ratio lever "
                                   "cells (LT3 |r_LT3 - r_v3| > 0.75%, U |r_v3 - 1| > 4.5%); LT3 expected UNDECIDED by construction",
                           "rms_max": 0.015, "max_abs": 0.04},
            "B-SKEW": {"statistic": "e_B = q_v3 / q_counted - 1 per (cell, GEMM), q = (dram_bytes_read - n x operand) / W_g, on the byte pages under the clock rule",
                       "rule": "HOLDS if every PT / PW (cell, GEMM) row has |e_B| <= 5% and their rms <= 3%, POOLED over OLMoE and Phi and "
                               "both GEMMs EXCEPT OLMoE w1 skewed rows (OPEN-L2SKEW, scored apart); per shape and per GEMM printed; "
                               "PT and PW only because they are the byte page's only skewed labels", **C6.B_SKEW},
            "OPEN-L2SKEW": {"what": "OLMoE w1 PT / PW rows, scored apart: OPEN (no model change follows from them)",
                            "prior": "v3 FAILS here (rental 5 read +8% at n 4 and +25% at n 16; (a) does not move a skewed cell)",
                            "rule": "v3: HOLDS / FAILS by the B-SKEW bar on these rows; ROWKEY (the rival at 39.5 MiB, priced in byte_predictions): "
                                    "BETTER if every row's |e| <= 5% and its rms <= 0.5 x v3's on the same rows, else NOT BETTER; 60 MiB printed"},
            "B-CA": {"statistic": "e_B on the w1 uniform and balanced rows, pooled over the byte pages; per shape printed",
                     "rule": "v3 HOLDS if every row's |e_B| <= 3%; v3_noca (q_pred_v3_noca) is EXCLUDED if its rms is at least 2x v3's and at least 3%",
                     **C6.B_CA},
            "E2-SKEW": {"statistic": "time from counted bytes under v3 (r6model.Pricer.price_counted, the page's A) against the timed page's "
                                     "same (model, n, label, arm) cell (page A)",
                        "rule": "HOLDS if rms <= 2%, worst <= 5% and |mean| <= 1.5%; the DECOMPOSITION ln(T_v3 / T_meas) = ln(T_v3 / T_c) + "
                                "ln(T_c / T_meas) printed per cell", **C6.E2_BAR},
            "calibration_line": "secondary, descriptive at 3 shapes: below 6 clusters NOT RESOLVED",
        },
        "noise": {"sigma_page": C6.SIGMA_PAGE, "lever_sigma": "0.18% only (review (h) 13); 0.11% and 0.078% printed in the design"},
        "labels": {"mixtral-8x7b": "BLIND-CALMODEL; balanced n 3, 6 SEEN replicates", "olmoe-1b-7b": "BLIND draws, IN-SAMPLE for (b); balanced n <= 9 SEEN",
                   "phi-3.5-moe": "BLIND; balanced n 2, 4 SEEN replicates; n 12 BLIND"},
        "scorer_ambiguities": AMBIGUITIES,
        "missing": "a missing page's cells are NOT SCORED; the nine skew pages carry nodrop=1 and no shared drop-group",
        "seen_data": ["the histograms' shapes are SEEN routing statistics", "h SEEN-FITTED; content_a SEEN-SELECTED", "sigma_page SEEN"],
    }


# --------------------------------------------------------------------------
# rental6-q1
# --------------------------------------------------------------------------

def _pairs(rows: list[dict], sel) -> list[dict]:
    d = {}
    for r in rows:
        if sel(r):
            d.setdefault((r["model"], r["page"], r["n"], r["label"], r["copies"]), {})[r["arm"]] = r
    out = []
    for (m, pg, n, lab, cp), v in sorted(d.items()):
        if "native" not in v or "shared" not in v:
            continue
        S, N = v["shared"], v["native"]
        out.append({"model": m, "page": pg, "n": n, "label": lab, "copies": cp, "q1page": pg.startswith("c"),
                    "cluster": f"{m}|n{n}", "native_path": S["native_path"],
                    "host_ok": S["host_ok"] and N["host_ok"],
                    "g_R": {R: r6(1e3 * ((S["ms"][R] - N["ms"][R]) - (S["ms"]["v3"] - N["ms"]["v3"]))) for R in M6.VARIANTS},
                    "gap_v3_us": r6(1e3 * (S["ms"]["v3"] - N["ms"]["v3"]))})
    return out


def q1() -> dict:
    rows = timed_cells()
    ctrl = ("uniform", "balanced")
    H_CELLS = {("phi-3.5-moe", 2), ("qwen1.5-moe-a2.7b", 2), ("jetmoe-8b", 4)}
    H_E1 = {("olmoe-1b-7b", 3), ("phi-3.5-moe", 2)}
    selH = lambda r: ((r["test"] == "Q1" and (r["model"], r["n"]) in H_CELLS)
                      or (r["test"] == "E1" and r["label"] in ctrl and (r["model"], r["n"]) in H_E1))
    H = [p for p in _pairs(rows, selH) if p["native_path"] == "block-scan" and p["host_ok"]]
    selP = lambda r: ((r["test"] == "Q1" and ((r["model"] == "jetmoe-8b" and r["n"] in (1, 2)) or (r["model"] == "phi-3.5-moe" and r["n"] == 1)))
                      or (r["test"] == "E1" and r["model"] == "mixtral-8x7b" and r["n"] == 3 and r["label"] in ctrl))
    Pall = _pairs(rows, selP)
    P = [p for p in Pall if p["host_ok"]]
    host_out = [f"{p['model']}/{p['page']}/n{p['n']}" for p in Pall if not p["host_ok"]]
    selSK = lambda r: (r["test"] == "E1" and r["label"] in SKEWED
                       and ((r["model"] == "olmoe-1b-7b" and r["n"] == 3) or (r["model"] == "phi-3.5-moe" and r["n"] == 2)))
    SK = _pairs(rows, selSK)
    qc = {cp: next(r for r in rows if r["test"] == "Q1" and r["model"] == "qwen1.5-moe-a2.7b" and r["n"] == 1 and r["arm"] == "shared" and r["copies"] == cp)
          for cp in COPIES}
    dead = {cp: sum(qc[cp]["dead_ctas"].values()) for cp in COPIES}
    ratio = dead[15] / dead[9]
    se_D = math.sqrt(2) * SIG_US["qwen1.5-moe-a2.7b"]
    power = {"device": "cpu", "dtype": "float64", "R": 200_000,
             "note": "registered on the CPU in float64; the build re-ran every point on MPS in float32 (max |diff| in CHANGES.md)"}
    for name, pairs in (("Q1-H", H), ("Q1-P", P)):
        for scen, kw in (("replicate sigma", {}), ("+ tau 1.3 us per (model, n)", {"tau_us": TAU_US}),
                         ("SEEN-anchored", {"anchor_us": SEEN_ANCHOR_US})):
            power[f"{name} | {scen} | v3 true"] = C6.q1_power(pairs, SIG_US, rivals=RIVALS_GAP, **kw)
        for truth in RIVALS_GAP:
            power[f"{name} | replicate sigma | {truth} true"] = {
                "P(v3 HOLDS)": C6.q1_power(pairs, SIG_US, rivals=(), truth=truth)["P(v3 HOLDS)"]}
    for x9 in (-6.6, -7.95):
        for truth in ("CALL", "DEAD"):
            power[f"Q1-C | x9 {x9} | board 1.0 + replicate 0.45 | {truth} true"] = C6.q1c_power(
                x9, truth, ratio, sig_x=SIG_US["qwen1.5-moe-a2.7b"], board=1.0, se_D=se_D)
    sens = {"what": "P(v3 HOLDS) on Q1-H (replicate sigma, v3 true) when Phi's and JetMoE's sd, which have no "
                    "replicate and are ASSUMED (Phi 1.0 us, JetMoE 0.45 us), are larger; Mixtral's SEEN sd is 2.04 us",
            "points": {}, "mps_float32_check": Q1H_SENS_MPS}
    for phi, jet in Q1H_SENS:
        sig = dict(SIG_US, **{"phi-3.5-moe": phi, "jetmoe-8b": jet})
        sens["points"][f"phi {phi} us, jetmoe {jet} us"] = C6.q1_power(H, sig, rivals=())["P(v3 HOLDS)"]
    pH = power["Q1-H | replicate sigma | v3 true"]["P(v3 HOLDS)"]
    pP = power["Q1-P | replicate sigma | v3 true"]["P(v3 HOLDS)"]
    pPs = power["Q1-P | SEEN-anchored | v3 true"]["P(v3 HOLDS)"]
    qp_status = ("DESCRIPTIVE" if (pP < 0.8 or pPs < 0.5) else "SCORED")
    return {
        "registered": REGISTERED, "name": C6.NAMES["q1"], "status": "REGISTERED BEFORE ANY PAGE", "design": DESIGN,
        "tool": "scorer scripts/scoring/rental6/score_q1.py; r6common.q1_rule, q1c_rule, q1_power, q1c_power",
        "pages": {"q9": "c9 balanced pages (same-board references; SEEN replicates)", "q15": "c15 balanced pages (SHARED BLIND; NATIVE SEEN: NATIVE declares E at any copies count, so its cells "
                                                                              "repeat the 2026-09-29 c9 NATIVE cells)",
                  "models": {m: list(v) for m, v in Q1.items()}, "files": _files([f"q1-{m}" for m in Q1])},
        "predictions": [r for r in rows if r["test"] == "Q1"],
        "statistic": {"g": "per (NATIVE, SHARED) pair on one page: g = (T_S - T_N)_meas - (T_S - T_N)_v3, us, T_v3 carried to the "
                           "page's own A (r6model.reprice); a page with no align_probe: its SHARED cells are NOT SCORED in Q1-H and Q1-P",
                      "se": "cluster-robust (CR1) on (model, n): se^2 = C / (C - 1) x sum_c (sum_i in c (g_i - mean))^2 / N^2",
                      "g_R": "rival R's per-pair (S - N)_R - (S - N)_v3 from the registered predictions (both carried to the page's A), "
                             "averaged over the SAME valid pairs"},
        "tests": {
            "Q1-H": {"noise_sensitivity": sens, "population": "CONTROL pairs only (uniform, balanced) whose NATIVE takes block-scan and that pass the host rule: "
                                   "Phi n 2 (E1 pages A, B, C and Q1 c9, c15), OLMoE n 3 (E1 A, B, C), Qwen1.5 n 2 (Q1 c9, c15), "
                                   "JetMoE n 4 (Q1 c9, c15: priced here, review (h) 16); skewed pairs are Q1-SK",
                     "pairs": H,
                     "rule": "v3 HOLDS iff |mean g| <= 1.5 us AND |mean g(c15 Q1 pairs) - mean g(c9 Q1 pairs)| <= 1.5 us (both sides "
                             "needed, else UNDECIDED); a rival is EXCLUDED iff |mean g - g_R| > 3 se and > 1.5 us; UNDECIDED below 6 pairs",
                     "rivals": list(RIVALS_GAP), "power_v3_replicate": pH,
                     "in_sample": "OLMoE n 3 pairs are IN-SAMPLE for h"},
            "Q1-P": {"population": "small-batch NATIVE pairs passing the host rule: Phi n 1 (Q1 c9, c15), Mixtral n 3 (E1 A, B, C controls)",
                     "host_bound_excluded": host_out,
                     "pairs": P, "status": qp_status,
                     "why": (f"DESCRIPTIVE: with JetMoE n 1, 2 out (host-bound by the registered rule), P(v3 HOLDS) is {pP} under "
                             f"replicate sigma and {pPs} at the SEEN anchors (Mixtral n 3 reads -8.03 us); the small-batch error "
                             "trends with n (SEEN: Mixtral +2.6, +1.0, -6.3 us at n 1-3; JetMoE n 3 +2.8; 8x22B n 3 -9.0), which "
                             "s_small cannot carry, and more cells do not fix it. The statistic and every rival's g_R are printed; "
                             "no verdict") if qp_status == "DESCRIPTIVE" else "scored by the Q1-H rule"},
            "Q1-SK": {"population": "skewed pairs (PT PW PW2 DW) at the Q1-H treads (OLMoE n 3, Phi n 2)", "pairs": SK,
                      "status": "OPEN, printed: mean g over skewed pairs minus mean g over the same pages' control pairs, per shape",
                      "seen_prior_us": SKEW_OFFSET_SEEN},
            "Q1-C": {"cell": "Qwen1.5 SHARED n 1 (NATIVE block-scan), c9 and c15 Q1 pages; x = g of that pair",
                     "statistic": "D = x15 - x9 (owner, review (e)): a board shift common to the two pages cancels",
                     "rivals": {"CALL": "D = 0 (a per-call cost at n = 1)",
                                "DEAD": f"D = (ratio - 1) x9, ratio = dead CTAs c15 / c9 = {dead[15]} / {dead[9]} = {r6(ratio)}",
                                "ZERO": "x9 = 0: printed only (SEEN x9 already reads -7.95 / -5.97 us)",
                                "OVERLAP": "partial host overlap at n 1 (rental 3): predicts what CALL predicts; printed as such"},
                     "dead_ratio": r6(ratio), "se_D_us": r6(se_D),
                     "rule": "the nearer of CALL and DEAD to D is SELECTED if the other is more than 2 se_D farther; else NOT SEPARATED; "
                             "positions are functions of the MEASURED x9 (review (h) 5); se_D = sqrt2 x 0.45 us (Qwen1.5 replicate, SEEN)",
                     "host_margin": "Qwen1.5 NATIVE n 1 is 339.6 us + F240 = 407 us against the 0.40 ms cut: the rule reads the REGISTERED "
                                    "prediction, so a faster board does not void the cell (the page's timer flag still can)"},
        },
        "power": power,
        "not_decided": ("no registered verdict separates v3 from v3_all (the block-scan path condition): on the Q1-H pairs (b) "
                        "applies under both (g_R v3_all = 0), and Q1-P, the one population where they differ (g_R v3_all about "
                        "-10 us), is DESCRIPTIVE; the condition stays SEEN-INFORMED, and Q1-P's printed g is the evidence"),
        "noise": {"sig_us": SIG_US, "label": "SEEN (rental-5 replicate pairs; Phi and JetMoE assumed)", "tau_us": TAU_US,
                  "seen_anchor_us": {f"{m}|n{n}": v for (m, n), v in SEEN_ANCHOR_US.items()}},
        "gates": "as rental6-skew (G1 with slips, 6 of 9 clean, G2 host rule + timer flag, G4 1705, G5); views ALL and CLEAN",
        "scorer_ambiguities": AMBIGUITIES,
        "labels": {"c9": "SEEN replicates (Phi, JetMoE, Qwen1.5 c9 pages exist, probe on)", "c15": "SHARED BLIND (no c15 SHARED page, probe on at c15 for the first time); NATIVE SEEN "
                                                                                  "(it declares E at any copies count: the 09-29 c9 NATIVE cells, rental 5 for Qwen1.5)"},
        "seen_data": ["h SEEN-FITTED", "the n = 1 exclusion SEEN-INFORMED", "replicate sigma and anchors SEEN"],
    }


# --------------------------------------------------------------------------
# rental6-secondk
# --------------------------------------------------------------------------

SECONDK_MODEL = "mixtral-8x7b-tp4"
#: OLMoE rental-4 counter c per k-step (SEEN, rental 4 SCORES.md part B'), as rental 5
C64 = {"w1": 346.2, "w2": 344.0}
C32 = {"w1": 245.2, "w2": 234.3}
K_OLMOE = {"w1": 2048, "w2": 1024}


def secondk_pred(model: str = SECONDK_MODEL) -> dict:
    from moe.spec import MODEL_CONFIGS
    cfg = MODEL_CONFIGS[model]
    K = {"w1": cfg.hidden_size, "w2": cfg.intermediate_size}
    out = {}
    for g in ("w1", "w2"):
        F = C4.F_CTA[g]
        S64, S32 = K[g] // 64, K[g] // 32
        u64 = S64 * C64[g] + F
        it = (S32 * C32[g] + F) / u64
        dF = (C32[g] - C64[g] / 2) * (K_OLMOE[g] // 32)
        cta = (S32 * C64[g] / 2 + F + dF) / u64
        sens = {}
        for f in (0.5, 1.5):
            Ff = F * f
            u = S64 * C64[g] + Ff
            sens[f"F x{f:g}"] = r6(100 * abs(((S32 * C32[g] + Ff) / u) / ((S32 * C64[g] / 2 + Ff + dF) / u) - 1))
        out[g] = {"K": K[g], "S64": S64, "S32": S32, "H_ITER": r6(it), "H_CTA": r6(cta), "dF_cycles": r6(dF),
                  "gap_pct": r6(100 * abs(it / cta - 1)), "gap_pct_F_sensitivity": sens}
    return out


def secondk() -> dict:
    return {
        "registered": REGISTERED, "name": C6.NAMES["secondk"], "status": "REGISTERED BEFORE ANY PAGE", "design": DESIGN,
        "tool": "scorer scripts/scoring/rental6/score_secondk.py; r4common.page_c; r6common.clock_rule",
        "pages": {"path": f"<date>-<card>-{SECONDK_MODEL}-<label>-r3-counters/lock1710/r3c-g64.json",
                  "labels": {"k64s4": [64, 4], "k32s4": [32, 4]},
                  "both_new": "BOTH pages run in rental 6 on the same board; the rental-1 k64s4 page is SEEN and is not scored",
                  "design_check": "each page's design (block_k, num_stages) must equal its label's; else NOT SCORED"},
        "statistic": {"c_page": "per (page, GEMM) c by r4common.page_c with ns = 4..9, F held at CAL (w1 520, w2 979 cycles), "
                                "max_dram_frac 0.5 (floor-bound cells only: DRAM read rate at most 0.5 x 4022 GB/s), over the cells "
                                "that pass the clock rule; fewer than 3 cells: NOT SCORED",
                      "r_meas": "u(k32) / u(k64), u = S c + F (cycles)", "e": "r_meas / r_pred(H) - 1"},
        "inputs": {"c64": C64, "c32": C32, "K_olmoe": K_OLMOE, "F_cal": dict(C4.F_CTA), "ns": list(C6.SECONDK_NS),
                   "max_dram_frac": C4.FLOOR_MAX_DRAM_FRAC,
                   "label": "c64 / c32 SEEN (OLMoE rental-4 B'); F CAL"},
        "hypotheses": {"H_ITER": "the BK 32 excess is per k-step: u(k32) = S32 c32 + F",
                       "H_CTA": "the same excess is per CTA: u(k32) = S32 c64 / 2 + F + dF, dF = (c32 - c64/2) x OLMoE's BK-32 steps"},
        "predicted_ratio": secondk_pred(),
        "rules": {"per_gemm": "within 3% = |r_meas / r_pred(H) - 1| <= 0.03 for the nearer H; that GEMM reads H; both beyond: NEITHER",
                  "verdict": "SELECTED <H> when both GEMMs read the same H; else UNDECIDED (readings printed)",
                  "noise": "counter noise about 0.3% against gaps of 16.9% (w1) and 22.4% (w2)"},
        "gates": {
            "V5": "each page's V5 (the slope gate, dram_counter_route) must PASS; the rental-1 k64s4 page passed at 0.9985",
            "clock": "each page by r6common.clock_rule: lock_gate.check_page PASS (V10, the lock in force, judged by the smi bracket, "
                     "null kernel and base twin), then each cell in [1640, 1710] MHz (out-of-band cells dropped); this replaces V10's "
                     "1705 cut, which the rental-1 page failed at 1659-1676 MHz (w2 fitted 1676, PRIVATE 1659-1663)",
            "pair_clock": "the two pages' median cell clocks must agree within 1% (cycles are compared across pages), else NOT SCORED",
            "V7": "V7 (the arm clock match) is not a gate on the cycle ratio; when a page's V7 FAILS, its c is read off its NATIVE "
                  "cells only (page_c on the native cells, at least 3), so no cross-arm clock mismatch enters the fit; the reading is "
                  "flagged 'V7: native only'",
            "views": "rental 2's two views as rental 5's secondk (occlaw.bytes_two_views)",
        },
        "missing": "a page missing or out of the view: NOT SCORED; drop-group k2 keeps the pair together",
        "labels": {"k64s4": "SEEN configuration (rental 1, this model, G 64)", "k32s4": "BLIND"},
        "seen_data": ["c64, c32 SEEN (OLMoE rental 4)", "F CAL", "the rental-1 k64s4 page SEEN (V5 0.9985; V7 and V10 failed)"],
    }


# --------------------------------------------------------------------------
# the 17 scorer ambiguities of design-r6-review (h), resolved
# --------------------------------------------------------------------------

AMBIGUITIES = {
    "1 Q1 c15-c9 clause": "over the Q1 pages' valid control pairs of the test (copies 9 and 15): mean g(c15) - mean g(c9), |.| <= 1.5 us; no se; both sides needed, else UNDECIDED",
    "2 se": "cluster-robust CR1 on (model, n) clusters, ddof via C / (C - 1); one cluster: iid sd / sqrt(N), ddof 1",
    "3 g_R": "the rival's registered per-pair (S - N)_R - (S - N)_v3, both carried to the page's A, averaged over the same valid pairs",
    "4 Q1-C se and distance": "se_D = sqrt2 x 0.45 us (registered); distance is 1-D on D = x15 - x9",
    "5 rival positions": "functions of the MEASURED x9: CALL D = 0, DEAD D = (ratio - 1) x9",
    "6 align statistic": "median over probe repeats per (arm, n), arm minus native, same page; histogram-independent (S3 test); a histogram cell takes A at its tread",
    "7 host rule": "rental 3's F240 rule on the REGISTERED v3 prediction (T + 0.067551 >= 0.40 ms) AND the page's timer flag; both drop",
    "8 drift-excluded repeat": "not clean (clean = neither lock-slipped nor drift-excluded)",
    "9 page G1 vs scorer G1": "R3's G1 PASS AND slipped rows <= 10%: a page failing either counts in ALL and is dropped in CLEAN",
    "10 G4 in [1695, 1705)": "the CELL is dropped; the page stays",
    "11 B-SKEW scope": "PT and PW (the byte page's only skewed labels); per (cell, GEMM) rows, POOLED over shapes and GEMMs except OLMoE w1 skewed (OPEN-L2SKEW); per shape and GEMM printed",
    "12 B-CA pooling": "pooled over the byte pages' w1 uniform / balanced rows; per shape printed",
    "13 lever sigma": "0.18% (rental 3 rep8): threshold 0.54%",
    "14 E1 mean": "pooled over all valid histogram cells, in each view; the registered verdict is rental 2's rule 3 combination",
    "15 second K": "within 3% = |r/r_H - 1| <= 0.03; ns 4..9; F CAL 520 / 979; max_dram_frac 0.5",
    "16 JetMoE n 4": "priced (rental6-q1 predictions); a block-scan control pair in Q1-H",
    "17 SEEN-table proxy": "rental6-v3's align_proxy block (r6model.align_proxy over the published reports)",
}


BUILDERS = {"v3": v3, "skew": skew, "q1": q1, "secondk": secondk}


def _txt_v3(d):
    s = d["seen_reproduction"]
    return [f"h = {d['h']['value_us']} us ({d['h']['label']})",
            "rental-5 E1-SKEW 72 cells, rms / mean / worst: " + "; ".join(
                f"{v} {x['rms']:.4f} / {x['mean']:+.4f} / {x['worst']:+.4f}" for v, x in s["rental5_E1_SKEW_72"].items()),
            f"8x7B CAL fitted cells under v3: rms {s['cal_8x7b_fitted_cells_v3']['rms']:.5f}"]


def _txt_skew(d):
    return [f"{m}: r_v3 - 1 {v['r_v3_minus_1_pct']} %; T_v3 {v['T_v3_us']} us" for m, v in d["summary"].items()] + \
           [f"rival {R}: rms gap {v['rms_pct']}%, worst {v['worst_pct']}%, lever cells {v['lever_cells_0.18']}" for R, v in d["rival_gaps_on_E1"].items()]


def _txt_q1(d):
    return [f"{k}: {v}" for k, v in d["power"].items() if isinstance(v, dict)] + \
           [f"Q1-P status: {d['tests']['Q1-P']['status']}"]


def _txt_secondk(d):
    return ["predicted u(k32)/u(k64): " + "; ".join(f"{g} H_ITER {v['H_ITER']:.4f} H_CTA {v['H_CTA']:.4f} (gap {v['gap_pct']:.1f}%)"
                                                  for g, v in d["predicted_ratio"].items())]


TEXT = {"v3": _txt_v3, "skew": _txt_skew, "q1": _txt_q1, "secondk": _txt_secondk}
