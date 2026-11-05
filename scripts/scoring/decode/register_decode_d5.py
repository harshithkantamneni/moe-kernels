#!/usr/bin/env python
"""Writes the decode stage's fifth addendum, docs/registered/2026-11-04-decode-addendum5-gh200.{json,txt}, and the new
blind's page lists (docs/registered/2026-11-04-blind5-pages/), before any page of the three new blind models: model v5d
(chosen on DS's and CAL2's pages; its constants DS's readings, mechanically), the coded gate (addendum 4's recorded
intent) and its reading on the published trees, and the new blind (its models, cells, the owner's bars, units, plan
lines, rental, deadline and cap). Every earlier decode registration stays byte-identical (pinned here), and so do v4d,
v4d' and every earlier decode scorer.

    python scripts/scoring/decode/register_decode_d5.py --write [--out-dir docs/registered]
    python scripts/scoring/decode/register_decode_d5.py --check

Every number is computed here from committed files (DS's committed readings, the published CAL2 and decode blind trees,
the registered pages, v5d and v4d') or is a typed-in record with its provenance (the build's CAL2-only alternatives and
its ablation, labelled; the GD5 offline lookup). `--check` recomputes and compares the JSON (code_pins excluded, as every
registration), the text and the pages; exit 1 on any difference.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d2common as D2  # noqa: E402
import d3common as D3  # noqa: E402
import d4common as D4  # noqa: E402
import d5common as D5  # noqa: E402
import dcommon as DC  # noqa: E402
import register_decode as RGD  # noqa: E402

REPO = DC.REPO
REG = REPO / "docs" / "registered"
V5D = D5.V5D
D5_CODE = ("scripts/r3_timing_model_v5d.py", "scripts/scoring/decode/d5common.py", "scripts/scoring/decode/gate_d5.py",
           "scripts/scoring/decode/finalise_decode_d5.py", "scripts/scoring/decode/score_blind_d5.py",
           "scripts/scoring/decode/register_decode_d5.py", D5.DS_READINGS_COMMITTED)
FROZEN = tuple(f"docs/registered/{D4.NAME}.{x}" for x in ("json", "txt"))
SKIP_TXT = {"code_pins", "pages", "units", "rows", "cells", "worst_cells"}
WORK = "scripts/scoring/decode/d5_build"
#: the build records (committed; d5_build/run.py --check recomputes every .out)
BUILD = tuple(f"{WORK}/{n}" for n in ("d5b.py", "run.py", "ladders.py", "call_forms.py", "alternatives.py", "every_path.py",
                                      "ablation.py", "weak.py", "ladders.out", "call_forms.out", "alternatives.out",
                                      "every_path.out", "ablation.out", "weak.out"))

DESIGN = ("v5d is chosen on DS's readings (the settling rental DS, addendum 4, published 2026-11-04) and CAL2's pages "
          "only; nothing is fitted on the earlier decode blind pages or on any page of the three new models. The form is "
          "the owner's brief of 2026-11-04 (both regimes per tile at DS's constants, the per-tile w1 rate, the call "
          "kernels by a registered rule on the model's shape, align by E with the measured slope); the open choices are "
          "made on DS's and CAL2's pages and listed with the alternatives tried (decisions D5-1 to D5-8). The gate is "
          "addendum 4's recorded intent, coded; it reads NO-GO on the published trees on both clauses. The owner closed "
          "the decode stage on 2026-11-04 as calibrated, not certified: the new blind is registered and NOT RUN, its "
          "plan is never written, and the NO-GO is not overridden")
#: the owner's decision after the gate reading (2026-11-04)
CLOSED = ("the owner, 2026-11-04, after the gate read NO-GO: the decode stage closes as calibrated, not certified; the new "
          "blind is not launched and the NO-GO is not overridden; the new blind stays registered and NOT RUN (no plan "
          "file, no page of the three new models); Mixtral-8x22B, Qwen2-57B-A14B and Granite-3.0-1B-A400M remain unseen")
GATE_OUT = "scripts/scoring/decode/gate_d5.out.txt"


def decisions() -> dict:
    lab = (f"COMPUTED by the committed build records under {WORK} (alternatives.py, every_path.py, call_forms.py, weak.py, "
           "ladders.py, ablation.py; their outputs *.out beside; run.py --check recomputes them)")
    return {
        "label": lab + "; every alternative scored on CAL2 (and DS) only unless a line says otherwise",
        "D5-1 the regimes (owner's brief)": (
            "on every decode tile the CORES rho tail is removed (the GEMM at bandwidth, its partial wave at its compute "
            "floor) and gemm = max(G_bw s, b / r_cta,t) + d b / r1_t, d = 1 when n > S, at DS's SUBWAVE r_cta,t and DRAIN "
            "r1_t (DS's registered fits, unchanged). Alternative tried: SUBWAVE re-read on DS's own ladders with one offset "
            "per (tile, GEMM) instead of one per ladder (r_cta 49.7, 27.5, 27.5, 56.0 GB/s; its rms on DS's rows 7 to 10% "
            "against the registered fit's 2 to 5%); CAL2 reads the same STOP (worst cell -10.67%). Not taken"),
        "D5-2 the paths": (
            "SUBWAVE, DRAIN and W1 act on the aligned paths only (small-batch, block-scan); the naive path keeps v4d''s GEMM "
            "price. Why: DS read no naive row (its GEMM readings take small-batch and block-scan cells only), and a naive "
            "GEMM's CTAs are not one slab each (a repeated expert's slab is served from L2). Alternative: every path (the "
            "first build): CAL2 rms 2.60% against 2.41%, both STOP on the same cell. Chosen on CAL2 and DS's row set. "
            "FORKING PATH, disclosed: the every-path variant had already been re-scored on the SEEN blind (3.52% pooled) "
            "before this choice; the choice was made on CAL2 alone and goes against that re-score (v5d reads 4.41%)"),
        "D5-3 W1": (
            "s_t = 3577.96 / c_t (DS's W1RATE) on the three larger tiles scales w1's bandwidth part only (DS read b / r_cta "
            "and b / r1 in counter time); fix R stays on 16x64x128w4s4 (DS re-read it at 0.917911 against 0.918488, a "
            "replicate). Alternatives: s_t on the whole w1 price, ceiling and drain included: CAL2 2.41%, worst -10.78%; "
            "w2 also at the counters' 3895.49 GB/s (DS assumed that rate for w2 and did not measure it): CAL2 3.28%, worst "
            "-12.44%. Not taken"),
        "D5-4 CALL": (
            "B_other (the call kernels' bytes at DRAM speed) is replaced by silu_and_mul a + g B / 1 MB and moe_sum a + h k "
            "+ g B / 1 MB, the coefficients by OLS on DS's NATIVE rows (69 per class, six models). The forms were chosen by "
            "leave-one-model-out rms on DS's six models (DS pages only; 16 candidates compared, 15 distinct as [1, T k I] duplicates [1, B]; a forking path on DS's pages): "
            "silu_and_mul [1, B] 0.529 us (against [1] 1.633, [1, I] 1.418, [1, I, B] 0.808 (the DS summary line), "
            "[1, T k] 1.674, [1, I, T k] 1.101); moe_sum [1, k, B] 0.201 us (against [1] 0.817, [1, B] 0.704, [1, k H] "
            "0.541, [1, k H, B] 0.509 (the DS summary line), [1, k] 0.387, [1, H, k] 0.437, [1, k, T H] 0.262, [1, k H, "
            "T H] 0.570). CAL2 with the DS summary-line forms: 2.36%, the same STOP. k: DS spans 2, 4 and 8, the new "
            "models' 2 and 8; B: Mixtral-8x22B's largest silu_and_mul B is 1.14 x DS's largest and Qwen2-57B's largest "
            "moe_sum B 1.75 x, both extrapolated along the fitted line"),
        "D5-5 ALIGN": (
            "DS's align points (small-batch at one loop iteration; block-scan, BLOCK) piecewise linear in E between the "
            "measured E and held at the nearest measured E outside them (never extrapolated), 0 on the naive path; plus "
            "SLOPE 0.526921 us x (iterations - 1) on the small-batch path. The new models' E: Mixtral-8x22B E 8 and "
            "Qwen2-57B E 64 are measured points (Qwen2-57B's block-scan at E 64 measured too); Granite-1B's small-batch "
            "E 32 is the DECL point (DECL TRANSFERS) and its block-scan E 32 is NOT measured: held at E 40's 8.704 us "
            "(linear extrapolation from E 40 and 64 would read 8.896 us). Alternatives: SLOPE's per-E intercepts in "
            "place of the one-iteration medians: CAL2 2.40%; no slope: 2.54%, worst -11.32%"),
        "D5-6 the CAL constants": (
            "T0_GR, s_naive and s_block refitted on CAL2's kept cells under v5d (v4d's registered fit and rho selection, "
            "score_cal_d3's rules); nothing else is fitted on CAL2"),
        "D5-7 the three weak DS readings (coordinator, 2026-11-04; COMPUTED in d5_build/weak.py, weak.out)": (
            "kept at DS's constants: no refit, no fallback, no term dropped (dropping one would be a further choice, and "
            "CAL2 prefers none of the drops: no drain on 32x64x128w4s3 2.45%, W1 off on 32x128x64w4s3 2.47%, against "
            "2.41%). How far each transfers to the new models, from their configs and registered page lists only (prices "
            "at the CAL2 refit's constants, no page of theirs): (1) SUBWAVE: r_cta binds on one JetMoE w2 row per tile "
            "(n 64, 128, 96, 64), so it is a one-model reading; under v5d it binds on 0 of the 143 new main cells and 0 of "
            "CAL2's 91 (the new models' single-wave GEMMs hold 128 to 512 CTAs, none for Qwen2-57B, against JetMoE's "
            "binding rows of 64 to 128; b / r_cta is at most 0.63 of the GEMM at bandwidth on Granite-1B, 0.40 on "
            "Mixtral-8x22B and 0.17 on Qwen2-57B, so r_cta would have to read 1.6, 2.5 and 5.9 times lower to bind), so "
            "the new predictions do not depend on it; with no ceiling anywhere every new prediction is unchanged. (2) W1RATE on 32x128x64w4s3 "
            "(2 ladders, SE 49 GB/s): c moved by one SE moves 5 cells per model by at most 0.80% (Mixtral-8x22B), 0.78% "
            "(Qwen2-57B), 0.37% (Granite-1B); dropping the scale there moves them by at most 5.2%. (3) DRAIN on "
            "32x64x128w4s3 (q 2.2 SE from 0, 5 crossing rows): it acts on 6 main cells each of Mixtral-8x22B and "
            "Qwen2-57B (none of Granite-1B); q set to 0 moves them by at most 5.9% and 6.0% (rms 1.4% per model), q + 2 SE "
            "by at most 5.3% and 5.4%. These 6 + 6 cells are where a weak DS reading can carry a new model's worst cell "
            "past 8% on its own; the blind scorer reports per cell, so a failure there is attributable to this reading"),
        "D5-8 choices informed by the diagnosis of the old blind pages (listed as such)": [
            "that the two regimes, the per-tile w1 rate and the per-model call kernels are model terms at all: the two "
            "offline diagnoses of 2026-11-04 read the earlier blind pages and chose what DS measures (addendum 4's "
            "disclosure)",
            "removing v4d''s CORES rho tail (the diagnosis showed it overcharged Phi's and JetMoE's single-wave GEMMs)",
            "replacing B_other by the measured call kernels (the diagnosis's per-call residual)",
            "every constant and every open choice above: DS's and CAL2's pages only"],
    }


def ablation() -> dict:
    lab = (f"COMPUTED by {WORK}/ablation.py (ablation.out) and every_path.py (every_path.out): each term alone and every term but one, CAL2 refitted "
           "each time; the SEEN blind re-score beside is POST-HOC on seen pages (a profile of each feature, never a "
           "choice)")
    return {"label": lab,
            "rows (CAL2 verdict, rms, worst | SEEN blind pooled rms; Phi, Qwen1.5, JetMoE, deepseek, Granite-3B rms)": [
                "v4d' (no term): GO 2.29% -6.35% | 11.40%; 9.5, 5.3, 19.5, 2.3, 13.4",
                "v5d (every term): STOP 2.41% -10.67% | 4.41%; 4.5, 1.9, 6.8, 1.8, 5.4",
                "regimes alone: STOP 2.51% -8.60% | 6.71%; 6.3, 4.4, 8.2, 2.6, 10.0",
                "W1 alone: GO 2.66% -7.30% | 11.69%; 9.7, 5.9, 19.4, 2.3, 14.4",
                "CALL alone: GO 2.20% -6.28% | 10.94%; 9.1, 4.6, 18.3, 2.4, 13.6",
                "ALIGN with SLOPE alone: GO 2.06% -5.16% | 8.24%; 7.8, 3.5, 14.4, 3.0, 8.2",
                "every term but the regimes: GO 2.25% -6.25% | 7.88%; 7.6, 3.3, 13.2, 2.2, 8.8",
                "every term but W1: STOP 2.86% -8.60% | 4.65%; 4.9, 2.7, 6.6, 3.2, 5.2",
                "every term but CALL: STOP 2.31% -10.38% | 4.54%; 4.8, 2.5, 6.9, 1.8, 5.2",
                "every term but ALIGN: STOP 2.21% -9.98% | 6.64%; 5.5, 3.6, 7.7, 1.8, 11.1",
                "the every-path build (D5-2's alternative): STOP 2.60% -10.67% | 3.52%; 2.4, 1.7, 5.1, 1.8, 5.2"],
            "reading (INFERENCE)": (
                "the regimes carry the re-score (11.4% to 6.7% alone) and are what breaks CAL2's worst cell: Mixtral-8x7B T "
                "128 A 4 (w2: 128 CTAs of 3.74 MB, one wave, one CTA per SM) runs 51 us slower than v5d prices it. DS's own "
                "ladders show the same cell (d5_build/ladders.py): Mixtral w2 T 128 A 4 sits 23.7 us over its A 8 row after "
                "bytes, Phi's 16.0 us; at a counter offset of 9.91 us (the median of DS's 26 DRAIN-ladder offsets) the "
                "128-CTA rows read 21.1 to 24.7 GB/s per CTA (Phi, Mixtral, JetMoE at A 8) while JetMoE's 64-CTA row reads "
                "34.0 (at SUBWAVE's own fitted per-ladder offsets, 28.0 to 30.9 and 46.7: the conclusion holds either way); DS's "
                "per-CTA ceiling fits one rate per tile (46.9 GB/s on 64x128x64w4s3, binding only on JetMoE's 64-CTA row), "
                "so it cannot price both. No CAL2-only choice in D5-1 to D5-5 (the regimes on) moves that cell under 8%. No "
                "variant tried reads GO on both clauses: the CAL-GO rows above (v4d', W1, CALL or ALIGN alone, every term "
                "but the regimes) re-score at 7.88% to 11.69%, and the one variant under 4.0% (every path, 3.52%) reads "
                "CAL STOP")}


GATE_LINES: list[str] = []


def gate_block(terms: V5D.Terms) -> dict:
    import gate_d5 as G
    cal2, blind = REPO / D4.CAL2_PUBLISHED / "results", REPO / D4.BLIND_PUBLISHED / "results"
    g = G.gate(REPO / Path(D5.DS_READINGS_COMMITTED).parent, cal2, blind, registered=terms)
    GATE_LINES[:] = G.lines(g)
    return {
        "rule (owner, 2026-11-04; recorded in addendum 4, coded here)": (
            "gate_d5.py prints one GO / NO-GO line; GO needs every clause: READINGS (DS's readings give this "
            "registration's v5d terms, no reading BLOCKING), TREES (both published trees reproduce their committed scores), "
            "CAL (clause 1: CAL2 under v5d reads GO by the addendum-3 stop rule on its kept cells) and RE-SCORE (clause 2: "
            f"the published decode blind re-scored under v5d at the CAL2 refit reads a pooled rms under "
            f"{D5.RESCORE_RMS_MAX:.1%} over the committed {D5.RESCORE_CELLS} cells). finalise_decode_d5.py runs the same "
            "evaluation and refuses unless it reads GO"),
        "clause 2 label": D5.WEAK_FLOOR,
        "the gate on the published trees at registration (COMPUTED)": {
            "output file": GATE_OUT + " (gate_d5.lines on the published trees; --check recomputes it)",
            "line": G.line(g), "verdict": g["verdict"], "why": g["why"],
            "CAL": {k: g.get("CAL", {}).get(k) for k in ("verdict", "E1_CAL_IN_SAMPLE", "bars", "constants", "rho",
                                                         "worst_cells")},
            "RE-SCORE (POST-HOC on SEEN pages; a weak sanity floor, never a test)": {
                k: (g.get("RESCORE") or {}).get(k) for k in ("verdict", "pooled", "per_model", "read_at")}},
    }


def blind_block() -> dict:
    return {"models": list(D5.NEW_BLIND), "bars": D5.BARS_OWNER, "units": D5.blind_units(), "drop_order": ["gr1"],
            "pages": {"dir": D5.PAGES5, "label": "the new blind's page lists: scripts/decode_synth.py's frozen generators "
                                                 "(main_cells, ctl_cells), unchanged; registered, never timed",
                      "files": {f"{m}-{k}.json": D5.reg_file5(m, k) for m, k in D5.PAGE_KEYS}},
            "plan_units": D5.blind_plan_units(), "plan": D5.BLIND5_PLAN,
            "status": "NOT RUN (owner, 2026-11-04): the gate reads NO-GO and the decode stage closes",
            "plan_written_by": "finalise_decode_d5.py --plan-out, which refuses on NO-GO: the plan file is never written",
            "driver": f"scripts/{D5.BLIND_DRIVER} (pinned)"}


def new_models(terms: V5D.Terms) -> dict:
    with D5.blind5():
        desc = sorted("/".join(k) for k in D2.excluded_keys(D5.NEW_BLIND))
        cells = {m: len(DC.page_doc(m, "main")["cells"]) for m in D5.NEW_BLIND}
        ctl = {m: len(DC.page_doc(m, "ctl")["cells"]) for m in D5.NEW_CTL}
    out = {m: {"footprint (COMPUTED)": D5.footprint(m), "timed decode pages (COMPUTED)": D4.no_decode_page(m)["verdict"],
               "main cells": cells[m], "ctl cells (GR1)": ctl.get(m, 0),
               "descriptive cells (second M-tile)": sum(1 for x in desc if x.startswith(m + "/"))} for m in D5.NEW_BLIND}
    return {"models": out, "main cells": sum(cells.values()), "descriptive cells": len(desc),
            "reach under v5d (COMPUTED)": D5.new_model_reach(terms),
            "GD5 offline (tuned config)": D5.gd5_offline(),
            "fit": "every footprint is under 5 GB of weights; the earlier blind's PREFLIGHT read 75.6 GB free on the GH200 "
                   "beside a 25.8 GB footprint (MEASURED)"}


def expected() -> dict:
    return {
        "label": "ESTIMATES (mine, after the gate reading above), not measured; the blind is NOT RUN, so none is ever tested",
        "the gate reads GO on these trees": "no: it reads NO-GO (COMPUTED above), on both clauses",
        "not run (the owner's decision); had it run, it would pass pooled at 3% / 8% / 1.5%": (
            "about 0.05: v5d's worst cells sit where the new models live (single-wave long-slab GEMMs at about one CTA per "
            "SM, as Mixtral-8x22B's w2, single-wave on 28 of its 45 aligned cells; Granite-1B in Granite-3B's call-dominated regime, Granite-3B "
            "re-scoring at 5.4%, with 24 align iterations and an unmeasured block-scan E)"),
        "had it run, at least one new model CERTIFIED at the owner's bars": (
            "about 0.25, most likely Qwen2-57B-A14B (E 64, k 8; the E 64 points measured; Qwen1.5 and deepseek re-score "
            "at 1.9% and 1.8%), least likely Mixtral-8x22B"),
        "cost not spent": "about $4.24 (72 unit-min, 111 min wall), $5.34 at the hard cap"}


def seen_text() -> dict:
    return {
        "what v5d read": ["DS's committed readings (ds.readings.json, read_ds.py on the published DS session)",
                          "the CAL2 session's pages and score (SEEN)"],
        "what the gate and the ablation read": ["the published decode blind pages (SEEN, read by both diagnoses; "
                                                "POST-HOC here, never a choice)"],
        "the new blind models": {m: D4.no_decode_page(m)["verdict"] for m in D5.NEW_BLIND},
        "their page lists": (f"registered here under {D5.PAGES5} from decode_synth's frozen generators: cell lists, not "
                             "measurements; no page of theirs has been timed, and addendum 4's 'no decode page' (a "
                             "*decode*pages scan) stays true"),
        "configs read (COMPUTED, not pages)": "their H, I, E and k (moe/spec.py) for the reach table and the CALL rule",
    }


def build() -> dict:
    rd = D5.committed_readings()
    terms = D5.terms_from(rd)
    r = D5.blind_rental()
    return {
        "registered": f"{D5.DATE}, before any page of Mixtral-8x22B, Qwen2-57B-A14B or Granite-3.0-1B-A400M",
        "name": D5.NAME,
        "status": ("REGISTERED BEFORE ANY NEW BLIND PAGE: the decode stage's fifth addendum: model v5d (frozen), the coded "
                   "gate and the new blind; the gate reads NO-GO on the published trees on both clauses (below); the decode "
                   "stage closes as calibrated, not certified (owner, 2026-11-04): the new blind is registered and NOT RUN; "
                   "every earlier decode "
                   "registration stays byte-identical (pinned here) and every registered verdict stands"),
        "design": DESIGN,
        "tool": (f"scripts/scoring/decode/gate_d5.py scripts/scoring/decode {D4.CAL2_PUBLISHED}/results "
                 f"{D4.BLIND_PUBLISHED}/results [out] (its output on the published trees: {GATE_OUT}); finalise_decode_d5.py and score_blind_d5.py refuse "
                 "on NO-GO and are not run"),
        "decision (owner, 2026-11-04)": CLOSED,
        "model v5d": {
            "module": "scripts/r3_timing_model_v5d.py (over v4d', byte-identical)",
            "form": V5D.describe(terms),
            "terms": terms.as_dict(),
            "terms source": (f"d5common.terms_from on {D5.DS_READINGS_COMMITTED} (DS's readings, addendum 4): SUBWAVE "
                             "r_cta and DRAIN r1 per tile, W1RATE's c_t per larger tile, the CALL rule's OLS on DS's "
                             "NATIVE call rows, ALIGN's points and SLOPE's slope; nothing fitted after the readings"),
            "CALL rule fit (COMPUTED)": {c: {k: v for k, v in x.items() if k != "models"}
                                         for c, x in D5.fit_call_rule(D5.call_rows(rd)).items()},
            "BLOCKING readings (COMPUTED)": D5.blocking(rd) or "none",
            "reduction (COMPUTED)": D5.reduction(terms),
            "decisions": decisions(),
            "ablation": ablation(),
        },
        "gate": gate_block(terms),
        "blind (owner, 2026-11-04, D4-1; registered in addendum 5)": blind_block(),
        "new blind design": {
            "new models": new_models(terms),
            "rules (frozen otherwise)": ("decodeB-blind's and addenda 2 and 3's: the power check first; E1-DEC pooled and "
                                         "per model without the second-tile DESCRIPTIVE cells; GD2 voids a model's GRR page "
                                         "(the two GR1 pages); G3-D shuffle voids; G3-D label DESCRIPTIVE, never a void; "
                                         "SIGMA-D and the knee rows DESCRIPTIVE; the rivals by rental 7's rule; the scope "
                                         "line d3common.SCOPE"),
            "bars": "the owner's (D4-1): rms 3%, worst 8%, |mean| 1.5%, per model and pooled, not tightened",
            "finalisation": ("finalise_decode_d5.py, mechanical: the gate (refuses unless GO), then finalise_decode.frozen "
                             "and .predict (frozen) under d5common.v5d and blind5 on the registered pages"),
            "scope line": D3.SCOPE,
        },
        "rentals": {
            "new blind": r | {"composition": "prelude 3 + calibrate 2 + three GRR main pages "
                                             + " + ".join(str(u["est_min"]) for u in D5.blind_units()[:3])
                                             + " + two GR1 ctl pages "
                                             + " + ".join(str(u["est_min"]) for u in D5.blind_units()[3:])},
            "status": "NOT RUN (owner, 2026-11-04): the gate reads NO-GO and the decode stage closes; nothing is rented",
            "launch precondition (never met)": "gate_d5.py prints GO; the blind plan's header would state it",
        },
        "expected (ESTIMATES, not measured)": expected(),
        "seen": seen_text(),
        "unchanged": ("the lock 1710 and G4 1705, the instruments, every registered page, the census rule, the tile source, "
                      "the card constants, v4d and v4d' (byte-identical), every pin of the earlier decode registrations, "
                      "the drivers, vm_run.sh, docs/LAMBDA.md, and every registered verdict (the decode blind FAIL and the "
                      "CAL2 GO under addendum 3)"),
    }


def text(d: dict) -> list[str]:
    x = d["rentals"]["new blind"]
    g = d["gate"]["the gate on the published trees at registration (COMPUTED)"]
    L = [f"REGISTERED {d['registered']}: the decode stage (a fifth addendum), {d['name']}", f"status: {d['status']}",
         f"design: {d['design']}", f"tool: {d['tool']}", ""]
    L.append(f"GATE at registration (COMPUTED): {g['line']}")
    L.append(f"decision: {d['decision (owner, 2026-11-04)']}")
    L.append(f"new blind rental (registered, NOT RUN): {x['units_min']} min of units, about {x['wall_min']} min wall with "
             f"{x['boot_setup_min']} min of boot and setup: ${x['cost_usd']:.2f} at ${D2.RATE}/h; deadline launch + "
             f"{x['deadline_after_launch_min']} min; guardian hard cap launch + {x['hard_cap_after_launch_min']} min "
             f"(${x['cost_at_hard_cap_usd']:.2f} at most)")
    b = d["blind (owner, 2026-11-04, D4-1; registered in addendum 5)"]
    L.append(f"new blind bars (owner, D4-1): rms {b['bars']['rms_max']:.0%} per model, worst cell {b['bars']['max_abs']:.0%}, "
             f"|mean| {b['bars']['mean_max']:.1%}")
    red = d["model v5d"]["reduction (COMPUTED)"]
    kept = red["naive cells keeping the v4d' GEMM prices with the terms on"]
    L.append(f"reduction: {red['verdict']} ({red['identical with every term off']} of {red['cells']} page cells identical "
             f"to v4d' with every term off; {kept} "
             f"of {red['naive cells']} naive cells keep v4d''s GEMM prices)")
    L.append("new blind plan units (registered; the plan file is never written: NOT RUN):")
    L += ["  " + ln for ln in b["plan_units"]]
    L.append("")
    for k, v in d.items():
        if k in SKIP_TXT or k in ("registered", "name", "status", "design", "tool", "decision (owner, 2026-11-04)"):
            continue
        _flat("", {k: v}, L)
    return L


def _flat(prefix, v, out, depth=0):
    if isinstance(v, dict):
        for k, x in v.items():
            if k in SKIP_TXT:
                out.append(f"{prefix}{k}: (in the JSON)")
                continue
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


def pins() -> dict:
    prev = json.loads((REG / f"{D4.NAME}.json").read_text())["code_pins"]
    paths = tuple(dict.fromkeys(tuple(prev) + FROZEN + D5_CODE + BUILD + tuple(D5.page5_path(m, k) for m, k in D5.PAGE_KEYS)))
    return {p: RGD.sha256(REPO / p) for p in paths}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=REG)
    a = ap.parse_args(argv)
    files = D5.page5_files()
    pdir = REPO / D5.PAGES5
    if a.write:
        pdir.mkdir(parents=True, exist_ok=True)
        for n, t in files.items():
            (pdir / n).write_text(t)
    doc = build()
    doc["code_pins"] = pins()
    jpath, tpath = a.out_dir / f"{D5.NAME}.json", a.out_dir / f"{D5.NAME}.txt"
    jtext = json.dumps(doc, indent=1, default=str) + "\n"
    ttext = "\n".join(text(doc)) + "\n"
    gtext = "\n".join(GATE_LINES) + "\n"
    if a.check:
        strip = lambda x: {k: v for k, v in x.items() if k != "code_pins"}  # noqa: E731
        same_j = jpath.exists() and strip(json.loads(jpath.read_text())) == strip(json.loads(jtext))
        same_t = tpath.exists() and tpath.read_text() == ttext
        same_p = all((pdir / n).exists() and (pdir / n).read_text() == t for n, t in files.items())
        extra = sorted(p.name for p in pdir.glob("*.json") if p.name not in files) if pdir.is_dir() else []
        no_plan = not (REPO / D5.BLIND5_PLAN).exists()
        same_g = (REPO / GATE_OUT).exists() and (REPO / GATE_OUT).read_text() == gtext
        for name, same in ((jpath.name, same_j), (tpath.name, same_t), ("pages", same_p and not extra),
                           ("no blind plan (NOT RUN)", no_plan), (Path(GATE_OUT).name, same_g)):
            print(f"{name}: {'same' if same else 'DIFFERS'}")
        if extra:
            print(f"pages not registered here: {', '.join(extra)}")
        return 0 if same_j and same_t and same_p and not extra and no_plan and same_g else 1
    jpath.write_text(jtext)
    (REPO / GATE_OUT).write_text(gtext)
    tpath.write_text(ttext)
    print(f"wrote {jpath}\nwrote {tpath}\nwrote {len(files)} pages under {D5.PAGES5}\nwrote {GATE_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
