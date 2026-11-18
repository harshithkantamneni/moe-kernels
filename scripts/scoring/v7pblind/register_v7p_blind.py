#!/usr/bin/env python
"""Writes (--write) or checks (--check) model v7's PREFILL blind registration,
docs/registered/2026-11-18-v7-prefill-blind-gh200.{json,txt}, its pages (docs/registered/2026-11-18-v7-prefill-blind-hist)
and its plan scripts/plans/rentalV7P-prefill-blind-2026-11.plan.

    python scripts/scoring/v7pblind/register_v7p_blind.py --write [--final --owner-go "<date>: <the owner's words>"] | --check

--write writes a DRAFT: the plan carries a DRAFT line that scripts/vm_run_v7p.sh refuses (and the guardian, which also
asks the JSON for "draft": false and the plan's sha256). --write --final refuses unless the DRAFT is committed at HEAD
and its freeze holds, every chosen model still has no prefill-tile page (the census over the sessions published now),
the owner's go is given (--owner-go, recorded verbatim), and the final document equals the committed DRAFT on every
block but the DRAFT marks, the status, the census and the freeze: the models, the pages, the cells, the predictions,
the bars, the rules, the minimum, the fix and the disclosures cannot move after the DRAFT (M2). It never rewrites a
page. --check rebuilds the document in the mode the committed JSON records (the census over the sessions it records)
and compares: the text, the pages and the plan byte for byte, the JSON with every float within 1e-9 relative.

Building it prices every cell under v7 at four reference pars and re-runs the gate's prefill numbers with the
registered pricer; it reads no page of the three models (none exists)."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v7pcommon as V  # noqa: E402

REPO = V.REPO
REL = 1e-9

REOPENING = (
    "THE THIRD REOPENING OF PREFILL, AGAINST A REGISTERED GATE NO-GO (owner, 2026-11-18). Prefill closed when round 5's "
    "gate FAILED (2026-10-28) and B3's blind FAILED (2026-11-05: E1-TILE FAILS, Phi-3.5-MoE rms 3.50%); it was reopened "
    "for model v7 (2026-11-17, the second reopening), whose registered gate (docs/registered/2026-11-17-v7-gate-gh200) "
    "read PREFILL: NO-GO: REFUSED: ModuleNotFoundError: No module named 'refit4' (scripts/scoring/v7/gate_v7.out.txt). "
    "The DESCRIPTIVE re-run with the fix (scripts/scoring/v7/prefill_rerun_v7.txt) reads NO-GO too: G1 FAILS (held-out "
    "means -1.99 to +1.66% against 1.0%: Qwen2-57B -1.50, Mixtral-8x7B +1.42, tp2 -1.99, Mixtral-8x7B C2 +1.66) and G2 "
    "FAILS (the gate's own pass estimate P(A) 0.2067 under 0.5; B 0.284, D 0.1885). The owner has nonetheless chosen ONE "
    "prefill blind of v7, every switch on, RR included, on unseen models: this registration OVERRIDES the gate's "
    "registered PREFILL NO-GO, knowingly, with the gate's own estimate printed beside it. It is a forking path the paper "
    "must disclose. It reverses nothing: B3's FAIL, round 5's gate FAIL, the v7 gate's PREFILL NO-GO (REFUSED) and the "
    "decode FAILs stand")
STANDS = [
    "round 5's gate: FAIL (2026-10-28, ROUND5.md; G1 max held-out mean +2.39 tp2, P(A) 0.124)",
    "B3 (docs/registered/2026-11-04-tileB3-gh200): E1-TILE FAILS (scripts/scoring/tileB3/tileB3.score.txt); Phi-3.5-MoE, "
    "Qwen1.5-MoE-A2.7B, JetMoE-8B and deepseek-v2-lite spent as blind models",
    "the v7 gate (docs/registered/2026-11-17-v7-gate-gh200): PREFILL: NO-GO: REFUSED (the registered reading; the "
    "re-run's NO-GO is DESCRIPTIVE)",
    "v7's decode blind (docs/registered/2026-11-17-v7-decode-blind-gh200): FAIL (E1-DEC pooled FAILS, no model "
    "certified); decode's earlier blinds FAIL",
]


# ================================================================== the document

def models_block(cen: dict) -> dict:
    per = cen["per model"]
    from moe.spec import MODEL_CONFIGS as MC
    return {
        "chosen": list(V.MODELS),
        "why": V.MODEL_CHOICE,
        "every model of moe/spec.py unseen at prefill (COMPUTED: no prefill-tile page, no prefill-size row)":
            [m for m, x in per.items() if x["unseen at prefill"]],
        "B3's instrument per model (COMPUTED: SHARED declares E x 9 experts; R3 refuses 1024 or more)":
            V.instrument_check(),
        "shapes": {m: {"E": MC[m].num_experts, "k": MC[m].top_k, "H": MC[m].hidden_size, "I": MC[m].intermediate_size,
                       "W_e_MB": V.r6(3 * MC[m].hidden_size * MC[m].intermediate_size * 2 / 1e6),
                       "weights_GB_at_9_copies_per_arm": V.r6(9 * MC[m].num_experts * 3 * MC[m].hidden_size
                                                              * MC[m].intermediate_size * 2 / 1e9)} for m in V.MODELS},
        "earlier timings of the chosen models (COMPUTED from the census; none at a prefill tile)": {
            m: {"other tiles, per session": per[m]["disclosure: its published records at other tiles, per session"],
                "merged.csv rows": per[m]["disclosure: merged.csv rows (fused_experts and grouped-GEMM timings)"],
                "compile-only censuses": per[m]["disclosure: compile-only censuses naming it"]} for m in V.MODELS},
        "earlier timings, read": {
            "granite-3.0-1b-a400m": ("R3-tile (32x64x64 w8 s4) launch-floor pages, rentals 1 to 3 (2026-10-01, 10-02, "
                                     "10-06, 1710 MHz); decode pages at the four decode tiles in v7's decode blind "
                                     "(2026-11-18 v7b session, the blind's FAIL); no prefill tile, no other card"),
            "granite-3.0-3b-a800m": ("R3-tile pages (2026-09-30 granite session; launch-floor rentals 1 and 2); decode "
                                     "pages at the four decode tiles in the decode blind (2026-11-04 decodeBlind "
                                     "session, FAIL); no prefill tile, no other card"),
            "qwen2-57b-a14b-tp8": ("R3-tile pages on GH200 (rentals 3 and 4, 32x64x64 w8 s4; rental 5, 32x64x32 w8 s4 "
                                   "and 32x64x64 w8 s4); registered decode pages of the decode domain blind (DRAFT, "
                                   "2026-11-18, decode tiles; none timed when this was built); a TP shard of "
                                   "Qwen2-57B-A14B, a CAL model of round 5 and of v7's gate (A2's pages; held-out mean "
                                   "-1.50% under v7); no prefill tile, no other card"),
        },
        "census (COMPUTED)": cen,
        "census table (COMPUTED)": V.census_table(cen),
        "GD5 offline (vLLM 0.27.1's tuned-config lookup)": V.gd5(),
    }


def cells_block(fp: dict) -> dict:
    specs = V.page_specs()
    per = {}
    for m in V.MODELS:
        x = [s for s in specs if s[0] == m]
        per[m] = {"pages": [s[3] for s in x], "treads": {f"G {s[1]}": list(s[2]) for s in x},
                  "cells": sum(6 * len(s[2]) for s in x),
                  "tokens per tread": {str(n): V.tokens_at(m, n) for s in x for n in s[2]}}
    return {
        "recipe": ("B3's (docs/registered/2026-11-04-tileB3-gh200 CELLS, B2's pages by scripts/skew_synth_tile.py): "
                   "per model a 128x128x64 w8 s3 G 1 page at the two smallest treads at which vLLM 0.27.1 runs that "
                   "tile at G 1 with n < 6 (M4's Hopper G-1 rule) and a G 16 page at the smallest G-16 tread n <= 9 "
                   "(the declared copies; at k 8 G 16 needs M / E > 128, n >= 9, so one tread); tokens n x 128 x E / k; "
                   "per tread PT, PW, PT2 and PW2 drawn from OLMoE's fitted shapes (skew_synth.shape_param6 and _draw, "
                   "pinned, unchanged; seed BASE + 10000 + 1000 x model index + 10 n + label index, BASE 20281118, "
                   "shuffle seed BASE), uniform (n x 128 rows each) and balanced (counts null); arms NATIVE and SHARED, "
                   "9 declared copies, slip-policy cell, g1-rule r7-median-run3; NATIVE cells scored, SHARED cells "
                   "carry no prediction (B2's instrument)"),
        "not in the recipe here (budget)": ("B3's 64x128x64 w8 s3 page (Phi-3.5 64g1, 6 cells), its 64x128 sag pages "
                                            "and gate, its three byte pages (B-BYTES, DESCRIPTIVE) and the sag repeat "
                                            "(DESCRIPTIVE): together about 30 min more; the 64x128 tile is NOT tested "
                                            "by this blind"),
        "per model": per,
        "cells": fp["n_cells"],
        "pages": {"dir": V.PAGES, "label": "this blind's page lists (B3's recipe); never timed",
                  "files": V.files_block()},
        "plan_units": V.unit_lines(),
    }


def rules_block() -> dict:
    return {
        "bars": dict(V.BARS, source=V.BARS_SOURCE),
        "statistic": ("e = T_v7 / T_meas - 1 per valid NATIVE cell, T_v7 the finalised prediction (v7 at the board's "
                      "par), T_meas the cell's r7-median-run3 p50 (B3's E1-TILE statistic)"),
        "verdict (registered)": V.CERT_RULE,
        "why this verdict": (
            "B3's registration decided the bar for 'every NEW blind test' (owner, 2026-11-04, its BAR block) and read it "
            "as a conjunction over its own models (E1-TILE); v7's other blind, registered under the same numbers, reads "
            "it per model: 'a model is CERTIFIED when its own bar and the pooled bar hold' (decodeB-blind's rule line, "
            "carried by addenda 5 and 6 and docs/registered/2026-11-17-v7-decode-blind-gh200 'certification'). One "
            "model, v7, with two regimes is read by one rule; B3's two views and rule 3 are kept because they are how "
            "a prefill page is read (tcommon.timed_page: ALL and CLEAN)"),
        "minimum": V.MIN_RULE, "minimum cells": V.MIN_CELLS,
        "voids": V.VOID_RULES,
        "B3's E1-TILE (DESCRIPTIVE)": V.B3_E1_TILE,
        "DESCRIPTIVE beside (no verdict)": ["B3's E1-TILE conjunction and its 2-pages-in-CLEAN certification",
                                            "B2's bar (2%, 5%, 1%) on the same cells",
                                            "round 5 (v7 with every switch off) on the same cells: RR's rung",
                                            "splits: 128 G 1 and 128 G 16, per model and class",
                                            "the bar without G3's per-stratum voids"],
        "claim if certified": ("v7 predicts vLLM's fused MoE prefill call at 128x128x64 w8 s3 (G 1 and G 16) on GH200 "
                               "at 1410 MHz and 700 W within 3% rms / 8% worst / 1.5% |mean| on the certified models' "
                               "registered cells; a fail closes prefill"),
        "scope": ("a certification covers v7's NATIVE time at 128x128x64 w8 s3 at G 1 and G 16 on the GH200 at 1410 MHz "
                  "and 700 W, on the registered cells only (54: 18 a model), each CERTIFIED model named. It does not "
                  "cover 64x128x64 w8 s3, decode, SHARED cells, any other lock, power limit, card or model, the byte "
                  "law, or the sag on another board"),
    }


def model_block(fp: dict, rep: dict, rerun: dict) -> dict:
    gate = V.gate_registration()
    return {
        "module": "scripts/r3_timing_model_v7.py (the prefill branch, V7.price_prefill), unchanged",
        "as frozen by": (f"the v7 gate registration {V.GATE_NAME} (its freeze: {len(gate['freeze']['files'])} files, "
                         f"constants_sha256 {gate['freeze']['constants_sha256']}: v7, round5_chain.py and the vendored "
                         "round 5 among them); unchanged since"),
        "switches": V.V7.V7.as_dict(),
        "branch": ("round 5 as B3 froze it (STAG-CUT w1 with W2 on w2, the K1 knee at round 4's SEEN constants (p 6.0, "
                   "S0 2.75), the clock form of F1 at the board's 128x128 sag par) with RR on w1 at G 1 (C 47.5 MiB, "
                   "sigma 0.08 at 128x128, W_c 264, 6 replicas, seed 7); at G 16 v7 is round 5 exactly"),
        "the fix (part of the frozen model)": {"statement": V.FIX_STATEMENT, "source (verbatim)": V.FIX_SOURCE,
                                               "applied by": "v7pcommon.apply_fix, before every pass estimate"},
        "on-board input": ("the 128x128 sag par of this board's OLMoE f128 and Mixtral-8x7B f128 pages (B3's estimator, "
                           "b3common.page_fit = the chain's r4.sag_page_fit; gate b3sag's registered range); no other "
                           "constant is read from the rental"),
        "finalisation (mechanical, after the rental, before any blind page)": (
            "score_blind_v7p.py: the board's validity and the gate records first; then the par from the two f128 "
            "pages by the repo's estimator and the frozen chain's, equal to each other and to the b3sag record (each "
            "page's sha256, b3common.PAR_RTOL); every cell priced by v7pcommon.price at that par; an audit hook refuses "
            "the run if finalisation opens a gaps-* path or a path naming a blind model (the census aside)"),
        "gate reproduction (COMPUTED with the registered pricer and the fix)": rep,
        "equals the committed re-run (scripts/scoring/v7/prefill_rerun_v7.json, every switch on)": rep == rerun,
    }


def estimate_block(rep: dict, wb: dict) -> dict:
    return {
        "the gate's own (registered method, B2's bar 2% / 5% / 1%, on B2's 84 power cells, v7's five held-out means)":
            rep["estimate"],
        "label": "ESTIMATE",
        "reading": ("the v7 gate's G2 asked P(A) >= 0.5 to send prefill to a blind; v7's P(A) is 0.2067 (B 0.284, D "
                    "0.1885): the gate's own method puts a pass at about one in five at B2's bar. This blind is "
                    "registered against that number"),
        "at the working-model bar (B3's method, b3common.pass_estimates, the same cells and means; DESCRIPTIVE)": wb,
        "working-bar reading": ("on B2's 84 cells (B3's models, not this blind's) the same method gives A "
                                f"{wb['A (5 models)']['E1-TILE']} for B3's conjunction and "
                                f"{min(v for k, v in wb['A (5 models)'].items() if k not in ('E1-TILE', 'pooled'))} to "
                                f"{max(v for k, v in wb['A (5 models)'].items() if k not in ('E1-TILE', 'pooled'))} per "
                                "model; it carries no per-call risk and was not computed on this blind's cells (the "
                                "noise model's byte sensitivities are B2's)"),
    }


def rental_block(r: dict, plan: str) -> dict:
    return dict(r, plan=V.PLAN, plan_sha256=hashlib.sha256(plan.encode()).hexdigest(),
                units=V.units(),
                driver=(f"scripts/{V.DRIVER}: scripts/{V.PARENT_DRIVER} (B3's, pinned) plus lines marked V7P: the gate "
                        "v7pcensus (scripts/v7p_gate.py)"),
                launcher=(f"{V.LAUNCHER}: {V.PARENT_LAUNCHER} with lines marked V7P: the v7p driver only, the DRAFT "
                          "refusal names this registration"),
                lock="1410 MHz, 700 W (C2's prelude, B3's), G4 1405 MHz",
                card="one gpu_1x_gh200 (Lambda)",
                guardian=("outside the repository (moe-kernels-work 2026-11-18 guardians/tV7P_*, state in state-v7p): "
                          "launches only when this registration is FINAL and its plan_sha256 is the committed plan's; "
                          f"deadline launch + {r['deadline_after_launch_min']} min, hard cap launch + "
                          f"{r['hard_cap_after_launch_min']} min (a watchdog that also stops the foreground's children), "
                          "terminate on a launch with no IP, terminate confirmed by the status helper"))


def disclosures(r: dict, fp: dict, cen: dict) -> list[str]:
    st = fp["RR stakes (v7 against round 5, %, at A2's par)"]
    return [
        REOPENING,
        ("POST-HOC ORIGIN: the owner chose this blind on 2026-11-18 after seeing v7's decode blind result (FAIL) and the "
         "decode domain map's design, with the gate's NO-GO and the re-run's numbers in view; the models were chosen "
         "after the census and the gate were read (MODEL_CHOICE), before any page of them"),
        ("v7 UNCHANGED: every switch on (RR and D6FIX; D6FIX has no prefill term), v7 as the gate froze it, round 5's "
         "prefill branch with its SEEN constants; no refit, no new constant; the only addition is the refit4 fix, "
         "which changes no price and is registered verbatim"),
        ("RR's constants were chosen after pricing variants on SEEN pages (design-v7 section 3; RR's own pages, "
         "2026-11-12); at G 16 v7 is round 5 exactly; RR moves these cells by "
         + "; ".join(f"{k} {v['mean']:+.3f}% ({v['min']:+.3f} to {v['max']:+.3f})" for k, v in st.items())
         + " at A2's par (COMPUTED)"),
        ("the per-call risk B3 disclosed applies unchanged (one per-call constant, 36.16 us, at every numel; the "
         "block-scan align not priced by numel): every cell here is block-scan (R3's dry run), at numel 4,096 to "
         "73,728 (k 8 x M), mostly past the 11,520 the align fit was measured to; the Granite cells are short calls "
         "(0.12 to 0.71 ms at A2's par), so a per-call error weighs more on them than on B3's (0.5 to 3 ms)"),
        ("these three models were priced offline while this kit was built (their v7 predictions, no measurement); "
         "Mixtral-8x7B-tp8 and DeepSeek-V3-tp8 trial cells were priced too and left out (MODEL_CHOICE; no page of any)"),
        ("Qwen2-57B-A14B-tp8 is a TP shard of Qwen2-57B-A14B, a CAL model whose held-out 128 G 1 mean under v7 is "
         "-1.50% (prefill_rerun_v7); the Granite models share no (E, H, I, k) with any CAL or B3 model (census)"),
        ("the 64x128x64 w8 s3 tile, B3's byte pages and its sag repeat are left out to keep the rental near 2 h; the "
         "blind tests v7 at 128x128x64 w8 s3 only"),
        (f"cost: {r['units_min']} unit-min, about {r['wall_min']} min wall: ${r['cost_usd']} at the inherited "
         f"d2common.RATE $2.29/h, ${r['cost_usd_at_list']} at Lambda's list price of 2026-11-18 ($2.59/h); deadline "
         f"launch + {r['deadline_after_launch_min']} min, hard cap + {r['hard_cap_after_launch_min']} min "
         f"(${r['cost_at_hard_cap_usd']} / ${r['cost_at_hard_cap_usd_at_list']})"),
        ("the gate's held-out models (OLMoE-1B-7B, Qwen2-57B-A14B, Mixtral-8x7B, Mixtral-8x22B, Mixtral-8x7B-tp2) are "
         "SEEN; B3's blind models are spent; DeepSeek-V3 has vLLM fused_experts timings at prefill sizes on H200 and "
         "A100 (merged.csv, 2026-08-22 to 2026-09-01) and, with its shards, cannot run B3's instrument (census)"),
    ]


def build(draft: bool = True, sessions: list[str] | None = None, fp: dict | None = None, rep: dict | None = None,
          wb: dict | None = None, owner_go: str | None = None) -> dict:
    fp = V.frozen_predictions() if fp is None else fp
    rep = V.gate_reproduction() if rep is None else rep
    rerun = V.committed_rerun()
    wb = V.working_bar_estimate(rep["means_pct"]) if wb is None else wb
    cen = V.census(sessions)
    plan = V.plan_text(draft)
    r = V.rental()
    status = ("DRAFT, NOT REGISTERED: the owner's go is not recorded; the plan carries a DRAFT line and "
              f"{V.LAUNCHER} refuses it" if draft else
              "REGISTERED BEFORE ANY PAGE of the three models, the owner's go recorded")
    doc = {
        "registered": V.DATE,
        "name": V.NAME,
        "draft": draft,
        "status": status,
        "THIRD REOPENING, GATE OVERRIDE": REOPENING,
        "the gate's own pass estimate (ESTIMATE)": {"P(A)": rep["estimate"]["A (5 models)"],
                                                    "B": rep["estimate"]["B (5 models)"],
                                                    "D": rep["estimate"]["D (O, Q, 8x22B, tp2; 8x7B G1 aside)"],
                                                    "bar of the estimate": "B2's 2% / 5% / 1% (the gate's G2)"},
        "every earlier registered verdict stands": STANDS,
        "design": ("the owner's decision of 2026-11-18: ONE prefill blind of model v7, every switch on, RR included, on "
                   "unseen models, B3's recipe and instrument, the working-model bar, pass or fail; then the stage stops "
                   "and is reported either way"),
        "tool": ("scripts/scoring/v7pblind/register_v7p_blind.py --write [--final] / --check (this file, the pages and "
                 "the plan), scripts/scoring/v7pblind/score_blind_v7p.py <session dir> <out dir> (finalise and score), "
                 f"{V.LAUNCHER} (the launcher), scripts/{V.DRIVER} (B3's driver plus V7P), {V.GATE_SCRIPT} (the census "
                 "gate), scripts/tileB3_gate.py sag (B3's sag gate, unchanged)"),
        "what": ("E1-TILE's statistic BLIND under model v7 at 128x128x64 w8 s3, G 1 and G 16: v7 with every switch on, "
                 "frozen before any page, on three models no prefill-tile page has timed, read per model at the "
                 "working-model bar"),
        "models": models_block(cen),
        "cells": cells_block(fp),
        "rules": rules_block(),
        "model v7 (frozen)": model_block(fp, rep, rerun),
        "pass estimate": estimate_block(rep, wb),
        "predictions": {
            "label": ("COMPUTED by v7 before any page of the three models at four boards' 128x128 pars (A2's, C2's, "
                      "C4's: B3's reference pars; B3's own board's: the par the gate's B3 information was read at); "
                      "the rental's own par finalises them mechanically"),
            "boards": {b: list(v) for b, v in V.REF_PARS.items()},
            "unpriced": fp["unpriced"], "n_cells": fp["n_cells"],
            "RR stakes (v7 against round 5, %, at A2's par)": fp["RR stakes (v7 against round 5, %, at A2's par)"],
            "fingerprint (the scorer recomputes it before reading anything)": V.compact(fp),
            "cells at A2's par": fp["boards"]["A2"]["cells"],
        },
        "rental": rental_block(r, plan),
        "disclosures": disclosures(r, fp, cen),
        "expected (ESTIMATE, judgment, not computed)": (
            "low: the gate's own estimate is P(A) 0.21 at B2's bar; at the working-model bar B3's method on B3's "
            f"cells gives about {wb['A (5 models)']['E1-TILE']} for a three-model conjunction and about 0.8 a model; "
            "here the Granite cells are short block-scan calls where the per-call constant weighs most, and v7's "
            "held-out means spread -1.99 to +1.66%, so each model's |mean| <= 1.5% is the binding clause; judged "
            "about 0.3 to 0.5 that the pooled bar and at least one model hold, under 0.25 that all three certify"),
        "unchanged": ("every earlier registration, page, verdict and pinned file; moe/spec.py; v7 and its gate "
                      "registration; round 5's vendored files; B3's driver, gates and rules; the 1410 lock and 700 W"),
        "freeze": V.freeze(),
    }
    if not draft:
        doc["owner's go"] = owner_go
    return doc


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


SKIP = ("registered", "name", "draft", "status", "THIRD REOPENING, GATE OVERRIDE", "the gate's own pass estimate (ESTIMATE)",
        "every earlier registered verdict stands", "design", "tool", "predictions", "freeze", "models")


def text(d: dict) -> list[str]:
    head = "DRAFT (not registered)" if d["draft"] else "REGISTERED"
    e = d["the gate's own pass estimate (ESTIMATE)"]
    L = [f"{head} {d['registered']}, before any page of Granite-3.0-1B-A400M, Granite-3.0-3B-A800M or "
         f"Qwen2-57B-A14B-tp8: model v7's prefill blind, {d['name']}", f"status: {d['status']}", "",
         d["THIRD REOPENING, GATE OVERRIDE"], "",
         f"THE GATE'S OWN PASS ESTIMATE: P(A) {e['P(A)']}, B {e['B']}, D {e['D']} ({e['bar of the estimate']}; v7's "
         "held-out means, prefill_rerun_v7)", "", "EVERY EARLIER REGISTERED VERDICT STANDS:"]
    L += [f"  - {x}" for x in d["every earlier registered verdict stands"]]
    L += ["", f"design: {d['design']}", f"tool: {d['tool']}", ""]
    if "owner's go" in d:
        L += ["owner's go: " + str(d["owner's go"]), ""]
    m = d["models"]
    L += ["MODELS", f"  chosen: {', '.join(m['chosen'])}", f"  why: {m['why']}",
          "  census (COMPUTED over every registered page list and every published session's JSON records):"]
    L += [f"    {x}" for x in m["census table (COMPUTED)"]]
    for k in ("every model of moe/spec.py unseen at prefill (COMPUTED: no prefill-tile page, no prefill-size row)",
              "shapes", "earlier timings, read", "GD5 offline (vLLM 0.27.1's tuned-config lookup)"):
        v = m[k]
        if k.startswith("GD5"):
            L.append(f"  GD5 offline: {v['verdict']}: {v['label']}")
        else:
            L.append(f"  {k}: {json.dumps(v, default=str)}")
    L.append("")
    for k, v in d.items():
        if k in SKIP or k == "owner's go":
            continue
        _flat("", {k: v}, L)
    p = d["predictions"]
    L += ["", f"PREDICTIONS: {p['label']}", f"  boards: {json.dumps(p['boards'])}",
          f"  unpriced: {p['unpriced']}; cells: {p['n_cells']}",
          "  model / page / n / label / v7 ms at A2, C2, C4, B3's pars / round 5 ms at A2's par (DESCRIPTIVE) / RR m_hat"]
    fpc = p["fingerprint (the scorer recomputes it before reading anything)"]
    for i, c in enumerate(p["cells at A2's par"]):
        ms = " ".join(f"{fpc[b]['ms'][i]:.6f}" for b in ("A2", "C2", "C4", "B3"))
        rr = "-" if c["RR"] is None else f"{c['RR']['m_hat']:.4f}"
        L.append(f"    {c['model']} {c['page']} n{c['n']} {c['label']} {ms} {c['round5_ms (DESCRIPTIVE, every switch off)']:.6f} {rr}")
    fr = d["freeze"]
    L += ["", f"FREEZE: {fr['statement']}", f"  gate registration: {json.dumps(fr['gate registration'])}",
          f"  own files ({len(fr['own'])}):"]
    L += [f"    {h}  {q}" for q, h in fr["own"].items()]
    return L


def close(a, b, path="") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        if list(a) != list(b):
            return [f"{path}: keys differ"]
        return [x for k in a for x in close(a[k], b[k], f"{path}/{k}")]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        return [x for i, (u, w) in enumerate(zip(a, b, strict=True)) for x in close(u, w, f"{path}[{i}]")]
    if isinstance(a, float) and isinstance(b, float):
        return [] if math.isclose(a, b, rel_tol=REL, abs_tol=1e-15) else [f"{path}: {a!r} != {b!r}"]
    return [] if (type(a) is type(b) and a == b) else [f"{path}: {a!r} != {b!r}"]


#: blocks a --final may change against the committed DRAFT: the DRAFT marks, the status, the freeze (the plan's bytes
#: change), the owner's go (added); within models only the census and its table (they read the sessions published
#: since); within rental only plan_sha256 (the plan loses its DRAFT line)
FINAL_MAY_CHANGE = ("draft", "status", "freeze", "owner's go")
CENSUS_KEYS = ("census (COMPUTED)", "census table (COMPUTED)",
               "every model of moe/spec.py unseen at prefill (COMPUTED: no prefill-tile page, no prefill-size row)",
               "earlier timings of the chosen models (COMPUTED from the census; none at a prefill tile)")


def strip(d: dict) -> dict:
    out = {k: v for k, v in d.items() if k not in FINAL_MAY_CHANGE}
    if isinstance(out.get("models"), dict):
        out["models"] = {k: v for k, v in out["models"].items() if k not in CENSUS_KEYS}
    if isinstance(out.get("rental"), dict):
        out["rental"] = {k: v for k, v in out["rental"].items() if k != "plan_sha256"}
    return out


GO_PAT = re.compile(r"^20\d\d-\d\d-\d\d\b.{10,}")


def final_refusals(doc: dict, draft: dict | None, owner_go: str | None) -> list[str]:
    """Why a --final document may not be written over the committed DRAFT (empty: it may): the DRAFT exists at HEAD and
    is a DRAFT, its freeze holds, the owner's go is given (a date and the words), every chosen model still has no
    prefill-tile page, and the final document equals the DRAFT outside FINAL_MAY_CHANGE and the census."""
    why = []
    if not owner_go or not GO_PAT.match(owner_go.strip()):
        why.append("no owner's go: --owner-go '<YYYY-MM-DD>: <the owner's words>' is required")
    if draft is None:
        return why + [f"no committed DRAFT of {V.REG_JSON} at HEAD"]
    if draft.get("draft") is not True:
        why.append("the registration committed at HEAD is not a DRAFT")
    bad = V.verify_freeze(draft)
    if bad:
        why.append("the committed DRAFT's freeze does not hold: " + "; ".join(bad[:5]))
    per = doc["models"]["census (COMPUTED)"]["per model"]
    why += [f"{m}: {per[m]['verdict']}" for m in V.MODELS if not per[m]["unseen at prefill"]]
    diff = close(strip(json.loads(json.dumps(doc, default=str))), strip(draft))
    if diff:
        why.append("the final document differs from the committed DRAFT: " + "; ".join(diff[:5]))
    return why


def write(root: Path, doc: dict) -> None:
    jpath, tpath, ppath = root / V.REG_JSON, root / V.REG_TXT, root / V.PLAN
    for p in (jpath, ppath):
        p.parent.mkdir(parents=True, exist_ok=True)
    jpath.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    tpath.write_text("\n".join(text(doc)) + "\n")
    ppath.write_text(V.plan_text(doc["draft"]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    ap.add_argument("--final", action="store_true", help="register for real (the owner's go recorded)")
    ap.add_argument("--owner-go", default=None, help="with --final: '<YYYY-MM-DD>: <the owner's words>'")
    ap.add_argument("--root", type=Path, default=REPO, help="where to write or read the files (tests)")
    a = ap.parse_args(argv)
    jpath, tpath, ppath = a.root / V.REG_JSON, a.root / V.REG_TXT, a.root / V.PLAN
    if a.write:
        if a.owner_go is not None and not a.final:
            print("REFUSED: --owner-go belongs to --final")
            return 2
        if a.root == REPO and not a.final:
            V.write_pages(a.root)
        doc = build(draft=not a.final, owner_go=a.owner_go)
        if a.final:
            why = final_refusals(doc, V.committed_draft(), a.owner_go)
            if why:
                print("REFUSED: " + "; ".join(why))
                return 2
        write(a.root, doc)
        print(f"wrote {jpath.name}, {tpath.name}, {ppath.name} ({'DRAFT' if doc['draft'] else 'final'})")
        return 0
    if not jpath.is_file():
        print(f"{jpath.name}: missing")
        return 1
    want = json.loads(jpath.read_text())
    draft = bool(want.get("draft", True))
    sessions = want["models"]["census (COMPUTED)"]["sessions scanned"]
    doc = json.loads(json.dumps(build(draft=draft, sessions=sessions, owner_go=want.get("owner's go")), default=str))
    ok = True
    bad = close(doc, want)
    print(f"{jpath.name}: {'same' if not bad else 'DIFFERS: ' + '; '.join(bad[:5])}")
    ok &= not bad
    texts = [(tpath, "\n".join(text(doc)) + "\n"), (ppath, V.plan_text(doc["draft"]))]
    texts += [(a.root / p, t) for p, t in V.page_texts().items()]
    for p, t in texts:
        same = p.is_file() and p.read_text() == t
        ok &= same
        print(f"{p.name}: {'same' if same else 'DIFFERS'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
