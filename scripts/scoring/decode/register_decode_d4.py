#!/usr/bin/env python
"""Writes the decode stage's fourth addendum, docs/registered/2026-11-04-decode-addendum4-gh200.{json,txt}, its pages
(docs/registered/2026-11-04-decode4-pages/: the six DS counter pages on SEEN models) and the DS plan, before any DS
page. A MEASUREMENT registration only (owner, 2026-11-04, D4-7): the units and cells, the stop rules, and what each
DESCRIPTIVE reading is and how it is computed from the pages. It registers no model form, no constant and no gate.
Every earlier decode registration stays byte-identical (pinned here), and so do v4d, v4d' and every earlier decode
scorer.

    python scripts/scoring/decode/register_decode_d4.py --write [--out-dir docs/registered]
    python scripts/scoring/decode/register_decode_d4.py --check

Every number is computed here from committed files (v4d''s geometry, the registered pages, the generated plan, the
published CAL2 census, the committed CAL2 score and finalisation) or is a typed-in input with its provenance (the
owner's decisions of 2026-11-04, labelled). `--check` recomputes and compares the JSON (code_pins excluded, as every
registration), the text, the pages and the DS plan; exit 1 on any difference.
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
import dcommon as DC  # noqa: E402
import register_decode as RGD  # noqa: E402
import register_decode_d3 as RG3  # noqa: E402

REPO = DC.REPO
REG = REPO / "docs" / "registered"
r6 = DC.r6
V4DP = D3.V4DP
D4_CODE = ("scripts/scoring/decode/d4common.py", "scripts/scoring/decode/read_ds.py",
           "scripts/scoring/decode/reproduce_d4.py", "scripts/scoring/decode/register_decode_d4.py", D4.DS_PLAN)
FROZEN = tuple(f"docs/registered/{D3.NAME}.{x}" for x in ("json", "txt"))
SEEN_INPUTS = (D4.CAL2_SCORE, "scripts/scoring/decode/cal2.score.txt", D4.FINALISED_D3, D4.BLIND_SCORE,
               "scripts/scoring/decode/blind.score.txt", "scripts/decode_synth.py", D4.CAL2_CENSUS)
SKIP_TXT = {"code_pins", "pages", "units", "rows", "cells"}

DESIGN = ("the decode blind FAIL's two offline diagnoses (" + D4.DIAG + "; " + D4.DIAG_RESIDUAL + ") and the owner's "
          "decisions of 2026-11-04. Kept from the earlier builds of this addendum (never registered): D4-2a a short "
          "DESCRIPTIVE settling rental DS on SEEN models; D4-5 the align-loop slope measured, not assumed 0. Superseded "
          "by D4-7 (never registered, never code): D4-2b v4d'' and its terms, D4-2c the new blind's pages and scorers, D4-3 "
          "the gate, finalisation and blind wrappers, D4-4 the coded launch gate, D4-6 the re-registered model readings "
          "(DRAIN, W1RATE, CALL, BLOCK as v4d'' terms, BLOCKING). D4-7 (owner, 2026-11-04, after the v3 build's POST-HOC "
          "preview): DS registers NO model form and no v4d'' constants; its registration is a measurement registration "
          "only (the units and cells, the stop rules, and what each reading is and how it is computed from the pages, all "
          "DESCRIPTIVE); every DS unit is nodrop; Phi T 96 A 5 and T 128 A 3 are added, and the ladders must identify both "
          "regimes (sub-wave few-CTA slow cells and wave-boundary drain) on every decode tile, the cheapest SEEN cells "
          "added where a tile lacks them; the v4d'' model, the gate and the finalise and score scripts for the new blind "
          "are removed, to be built in the next round after DS; the future gate is recorded as intent only. D4-8 (owner, "
          "2026-11-04, after the v4 build showed every expert of Phi T 96 A 5 and T 128 A 3 past BLOCK_M): those two cells "
          "are replaced by Phi T 80 A 5, Phi T 80 A 6, OLMoE T 128 A 16 and OLMoE T 128 A 32, a third DRAIN model on "
          "32x128x64w4s3 and 64x128x64w4s3 (A 32 gives OLMoE a crossing row past one boundary, as JetMoE's and "
          "Qwen1.5's), each checked for rows per expert against BLOCK_M and its wave structure; and DS runs a tile "
          "census of its own with CAL's census gate: the reader's occupancy stop rule reads DS's census, CAL2's census "
          "becomes a reported cross-check, and a difference between the two is a STOP for the per-CTA readings")


def readings_text() -> dict:
    c = f"at least {D4.MIN_LADDERS} qualifying ladders from at least {D4.MIN_MODELS} models"
    return {
        "every reading": ("DESCRIPTIVE: computed from DS's pages by the rule below, registered before any DS page; none is "
                          "a model term, none is a STOP, none is a bar, and none changes a registered verdict. A reading "
                          "that does not clear its IDENTIFIED rule reads NOT IDENTIFIED and is reported "
                          "(read_ds.py's NOT_IDENTIFIED list); nothing falls back. Rows: the six DS pages' NATIVE arms "
                          "(DECL arms in ALIGN and SLOPE only), cells on the small-batch or block-scan path with no expert "
                          "past BLOCK_M (GEMM readings), refused cells and pages that fail PAGE left out, tiles whose CENSUS "
                          "fails left out, and no per-CTA reading (LADDERS, SUBWAVE, DRAIN, W1RATE) under a census STOP. "
                          "Per GEMM row: n live CTAs, S = 132 x the census occupancy (396 on 16x64x128w4s4 "
                          "and 64x128x64w4s3, 528 on 32x64x128w4s3 and 32x128x64w4s3), b = v4d''s bytes per CTA ((the "
                          "GEMM's reads + the output tile x n) / n), F = the GEMM's compute floor, B = the counted "
                          "dram__bytes_read, L = the counter page's gpu__time_duration in us, Dc = B / c (c = W1RATE's c_t "
                          "on the row's tile for w1; 3895.49 GB/s, the counters' MEASURED rate, fix R, for w2, its only "
                          "rate), X = L - max(Dc, F). NO FALLBACK (owner, 2026-11-04; this rule covers every reading): "
                          "w1's bytes are priced only at W1RATE's c_t on the row's tile, never at 3895.49 GB/s or any other "
                          "fallback. On a tile where W1RATE reads no rate, a w1 row is not priced (Dc and X none): SUBWAVE "
                          "reads NOT IDENTIFIED there when its fitted ladders hold a w1 row, DRAIN when its mixed ladders "
                          "do, and LADDERS shows the w1 rows unpriced; W1RATE takes no c, and CALLS, ALIGN, BLOCK and SLOPE "
                          "price no bytes. A LADDER "
                          "is one (model, T, GEMM); its rows share one offset c0 (the counter page's per-kernel time outside "
                          "the stream)"),
        "LADDERS (DESCRIPTIVE; the GEMM ladders on both sides of one wave and within a single wave)": (
            "per decode tile every ladder's rows by n: n, S, the wave class (sub-wave n <= S, crossing S < n < 2S, "
            "multi-wave n >= 2S), b, B, L, Dc, F and X; and which ladders are MIXED (a row with n <= S and a row with n > S: "
            "both sides of one wave) and which are SUBWAVE-qualifying (within one wave: two or more sub-wave rows whose n "
            f"span a factor of {D4.SUBWAVE_SPAN:g} or more, every one with b >= {D4.LONG_SLAB_BYTES / 1e6:g} MB; its "
            "smallest n is its few-CTA row). No fit"),
        "SUBWAVE (DESCRIPTIVE; few-CTA long-slab cells within a single wave)": (
            "rows: the sub-wave rows (n <= S) of every ladder holding two or more. FORMULA per decode tile t: r_cta,t "
            "minimises sum ((c0 + max(Dc, F, b / r) - L) / L)^2 over those rows, every c0 free (weighted least squares at "
            "each r), r over the registered grid: no ceiling, then 10 x 1.02^j GB/s for j = 0 to 232 (10 to 990 GB/s), "
            "ties to the first minimum in that order. r_cta,t is the per-CTA streaming rate a GEMM of few long-slab CTAs "
            f"runs at when its CTAs cannot fill the DRAM pipe. IDENTIFIED with {c} (SUBWAVE-qualifying); else NOT "
            "IDENTIFIED and not fitted. Reported beside: the binding rows (b / r over max(Dc, F)), the offsets, the "
            "relative rms at the fit and with no ceiling"),
        "DRAIN (DESCRIPTIVE; the wave-boundary drain)": (
            "rows: every row of the tile's MIXED ladders; d = 1 when n > S (the GEMM crosses a wave boundary), else 0. "
            "FORMULA per decode tile t: q_t minimises sum ((c0 + max(Dc, F, b / r_cta,t) + d b q - L) / L)^2, every c0 "
            "free (closed-form weighted least squares), r_cta,t SUBWAVE's reading on the tile (no ceiling when it reads "
            "none or is NOT IDENTIFIED); NO FALLBACK as stated under every reading; a q_t at or under 0 is set to 0 (no drain) and the offsets refitted; r1_t = 1 / "
            f"q_t GB/s, the rate of the one slab per CTA past the boundary. IDENTIFIED with {c} (MIXED). Reported beside: "
            "q's standard error, the offsets, the relative rms at the fit and with no drain"),
        "W1RATE (DESCRIPTIVE; the per-tile w1 byte-rate pairs)": (
            "rows: the w1 rows with n > S on each decode tile, past at least one wave boundary (a multi-wave row, n >= 2S, "
            "crosses more than one, so the ladder's offset holds a boundary cost only as far as its rows cross the same "
            "number of boundaries, and any difference falls into u_t). A ladder is one (model, T); it is usable with two or more rows whose B span at "
            f"least {D4.W1RATE_MIN_MB:g} MB (fix R's cut). FORMULA per tile t: L = c0_ladder + B u_t by ordinary least "
            "squares over the usable ladders' rows, c_t = 1 / u_t the counters' marginal w1 rate (GB/s); beside, its ratio "
            f"to v4d's 3577.96 GB/s. IDENTIFIED with {c} (usable). On 16x64x128w4s4 it re-reads fix R (0.918488, "
            "MEASURED 2026-11-03), a replicate; fix R itself is unchanged"),
        "CALLS (DESCRIPTIVE; silu_and_mul, moe_sum and the align kernel per call, per model)": (
            "the wide kernel filter (moe_align_block_size* and count_and_sort_expert_tokens* as align; act_and_mul* and "
            "silu* as silu_and_mul; moe_sum* as moe_sum; kernel replay, --cache-control all). rows: every NATIVE cell's "
            "class time (gpu__time_duration summed per call, the median over the 3 measured calls). Per model and class: "
            "the median, minimum and maximum us over its cells and the median per T. Beside, per call class, one summary "
            "line us = a + b x + g B / 1 MB by ordinary least squares over every NATIVE row (silu_and_mul x = I, B = T k 3 "
            "I x 2 bytes; moe_sum x = k H, B = (T k H + T H) x 2 bytes; H, I, k from the model's config), computed with "
            f"rows from at least {D4.CALL_MIN_MODELS} models holding at least {D4.CALL_MIN_MODELS} distinct x. The line "
            "summarises DS's pages; it is not a model term and not a rule for any other model"),
        "ALIGN (DESCRIPTIVE; the align kernel across E)": (
            "rows: every (cell, arm) with an align-class time, at the declared E (E on a NATIVE arm, E x C on a DECL arm: "
            "JetMoE C 4 and Phi C 2 declare 32, Mixtral C 2 declares 16) and the path vLLM takes at that E. Small-batch "
            "points: per E the median over NATIVE rows at one loop iteration (ceil(numel / max(E, 32)) = 1): E 8, 16, 40, "
            f"60, 64; DECL's E 32 enters only when DECL TRANSFERS (Mixtral's DECL E 16 median within {D4.DECL_TRANSFER_US:g} "
            "us of Phi's NATIVE E 16 median), else the DECL points are dropped. Block-scan points: BLOCK"),
        "BLOCK (DESCRIPTIVE; the block-scan constants per E)": (
            "per declared E the median align-pair ncu time over NATIVE block-scan rows (Granite-3B T 128 A 40 at E 40, "
            f"OLMoE T 128 A 64 at E 64). IDENTIFIED with points at {D4.BLOCK_MIN_E} or more E"),
        "SLOPE (DESCRIPTIVE; the align slope)": (
            "rows: the small-batch align rows ALIGN reads (NATIVE, and DECL when DECL TRANSFERS), each at its declared E "
            "and its loop iterations ceil(numel / max(E, 32)). FORMULA: align_us = a_E + s (iterations - 1), one intercept "
            "per declared E and one slope s, by ordinary least squares over every row of an E holding two or more "
            f"iteration counts. IDENTIFIED with {D4.SLOPE_MIN_E} or more such E. Beside: s over its OLS standard error "
            f"(DIFFERS FROM 0 at |t| >= {D4.SLOPE_T_CUT:g}, else NOT DISTINGUISHABLE FROM 0), the per-E slopes over every "
            "arm and their median, the iterations read (DS spans 1 to 24: Granite-3B T 120 A 40)"),
    }


def stop_rules() -> dict:
    return {
        "prelude (the D3 driver, unchanged)": (
            "the session STOPS before any page unless: the checkout is clean, passwordless sudo works, the GPU is idle, "
            "the supported clocks list 1710 MHz, persistence mode reads Enabled, and the GPU power limit set with -pl 700 "
            "-sc 0 reads back 700 W within 1 W"),
        "lock (the D3 driver, unchanged)": (
            "every DS page runs under locked_run at the 1710 MHz lock: the lock is set and read back from clocks.sm (two "
            "tries) before the counter route starts, else the unit exits 2 and writes no page; the page sits under "
            "lock1710 and the reader reads only that path (score_cal.counter_page). G4 (1705 MHz) is a timed-page gate; "
            "DS has no timed page"),
        "POWER (the reader, addendum 2's rule)": (
            "read_ds.py reads nothing unless exactly one session power-set.txt reads a GPU Current Power Limit of 700 W "
            "within 1 W"),
        "PAGE (the reader, per DS page)": (
            "a counter page is read only when its manifest's page_sha256 is the registered DS page's, its V_ALIGN_grid gate "
            "reads PASS (every GEMM launch's grid is the child's readback), its kernel filter is wide, and its software "
            f"versions are the CAL2 session's ({', '.join(f'{k} {v}' for k, v in D4.CAL2_VERSIONS.items())}; python on "
            "major.minor, its patch level recorded and printed, DESCRIPTIVE; torch, triton and vllm exact); else none of "
            "its rows enters any reading. Refused cells are left out"),
        "ROW SANITY (the reader, per row)": (
            "a GEMM row is refused, left out of every reading and listed (read_ds.py's REFUSED ROWS) unless its ncu time "
            "L and counted bytes B are finite and positive and B lies within "
            f"{D4.B_BAND[0]:g} to {D4.B_BAND[1]:g} x n b (n live CTAs x v4d''s bytes per CTA; a factor of 2 either way, a "
            "garbled-row guard, not a fit: CAL2's SEEN counter pages read 0.94 to 1.00, POST-HOC), and its implied DRAM "
            f"rate B / L is at most {D4.RATE_CEILING_GBPS:g} GB/s (owner, 2026-11-04: the GH200's HBM3 spec peak; no DRAM "
            "read runs faster, so a row past it holds garbled bytes or a garbled time. The published decode counter pages "
            "never exceed 3,768 GB/s, MEASURED, 6.3% under the peak; W1RATE's c_t above it reads NOT IDENTIFIED). A call or align class "
            "time that is not finite and positive is refused and listed the same way. A fit whose result is not finite "
            "reads NOT IDENTIFIED"),
        "CENSUS GATE (the reader, on DS's own census; owner, 2026-11-04, D4-8)": (
            "DS runs its own tile census (the plan's tilecensus unit: the pinned tile_census.py, compile only, the four "
            "decode tiles on the six DS models, nodrop, as CAL2's unit). The census gate CAL used (decodeC-cal's census "
            "rule, the tile stage's build-tile-review 9) reads it: exactly one DS census file "
            f"(<date>-<card>-tile-census-<model>-{D4.DS_LABEL}/tile-census.json), and no MMA path mismatch in it; a missing "
            "census or a path mismatch is a STOP for the per-CTA readings (LADDERS, SUBWAVE, DRAIN, W1RATE), never a "
            "fallback to CAL2's census"),
        "CENSUS (the reader, per decode tile, on DS's own census)": (
            "S comes from the census occupancy. DS's census must read, for every DS model's w1 and w2 on the tile, the "
            "occupancy S uses (3 CTAs per SM on 16x64x128w4s4 and 64x128x64w4s3, 4 on 32x64x128w4s3 and 32x128x64w4s3); "
            "else LADDERS, SUBWAVE, DRAIN and W1RATE read NOT READ on that tile"),
        "CENSUS CROSS-CHECK (the reader; reported, not gating; owner, 2026-11-04, D4-8)": (
            f"CAL2's published tile census ({D4.CAL2_CENSUS}; MEASURED 2026-11-03) is read under the same occupancy rule and "
            "reported, not gating. DS's census is compared with CAL2's entry by entry for every DS model, decode tile and "
            f"GEMM (the config and {', '.join(D4.CENSUS_FIELDS)}): SAME, or DIFFERS with every difference listed. DIFFERS "
            "is a STOP for the per-CTA readings (LADDERS, SUBWAVE, DRAIN, W1RATE), reported with the differences; CALLS, "
            "ALIGN, BLOCK and SLOPE do not take S and still read"),
        "G-BAND (not a DS rule)": (
            "A2's G-BAND gates the sag estimator's f128 counter clocks on the tile and CAL rentals; DS fits no sag "
            "estimator, takes no f128 page, and the D3 driver has no gate step. It does not apply to DS"),
        "budget (the driver and the guardian, unchanged)": (
            "a unit whose estimate is over the minutes left before the deadline is SKIPPED; every DS unit is nodrop, so "
            "nothing is dropped early; the guardian terminates at the hard cap; a step after which the card cannot be "
            "made safe (hygiene) STOPS the session"),
        "never": ("no DS reading is a STOP (only the stop rules above stop anything); a reading that does not identify "
                  "is reported NOT IDENTIFIED"),
    }


def measurement_block() -> dict:
    added = D4.ds_added()
    return {
        "decision (owner, 2026-11-04, D4-7)": (
            "DS registers no model form and no v4d'' constants. Its registration is a measurement registration only: the "
            "units and cells; the stop rules; and what each reading is and how it is computed from the pages, all "
            "DESCRIPTIVE"),
        "the model form": D4.MODEL_AFTER_DS[0].upper() + D4.MODEL_AFTER_DS[1:],
        "disclosure (plain words)": D4.SEEN_DISCLOSURE[0].upper() + D4.SEEN_DISCLOSURE[1:],
        "not registered here": ["any decode model form (v4d'' of the earlier builds is removed)", "any constant",
                                "any launch gate in code", "the new blind's pages, plan, finalisation or scorer"],
        "cells added (COMPUTED)": added,
        "the cells D4-8 replaced (COMPUTED)": {
            "cells": [D4.cell_check(m, T, A) for m, T, A in D4.DS_REPLACED],
            "why": ("Phi T 96 A 5 (32x128x64w4s3, BLOCK_M 32) holds 38 to 39 rows per expert and T 128 A 3 "
                    "(64x128x64w4s3, BLOCK_M 64) 85 to 86: every expert is past BLOCK_M, so both fell outside the GEMM "
                    "readings' row rule and entered CALLS and ALIGN only; their w1 GEMMs run two M-tiles per expert and "
                    "cross a wave (n 1000 and 600). D4-8 replaces them")},
        "the owner's four cells of D4-8 (plain words; cells added holds the COMPUTED checks)": (
            "Phi T 80 A 5 holds 32 rows per expert against BLOCK_M 32 (none past) with w1 n 500 within one wave (S 528), "
            "and Phi T 80 A 6 holds 26 to 27 rows with w1 n 600 across one boundary (S < n < 2S); together they make "
            "phi-3.5-moe/T80/w1 MIXED on 32x128x64w4s3 (b 1.06 MB), a third DRAIN model there beside JetMoE and Qwen1.5. "
            "OLMoE T 128 A 16 holds 64 rows per expert against BLOCK_M 64 (none past), block-scan, w1 and w2 n 256 within "
            "one wave (S 396), and a second block-scan row at E 64. OLMoE T 128 A 32 holds 32 rows per expert (block-scan) "
            "with w1 and w2 n 512 past one boundary (S < n < 2S), as JetMoE's and Qwen1.5's crossing rows on that tile; "
            "with T 128 A 64 (n 1024, 2.59 waves) they make olmoe-1b-7b/T128/w1 and /w2 MIXED on 64x128x64w4s3 (b 0.55 "
            "and 0.28 MB), a third DRAIN model there beside JetMoE and Qwen1.5"),
        "DS's own census (owner, 2026-11-04, D4-8)": (
            f"{D4.census_line()}: about {D4.CENSUS_EST} unit-min, read by the census gate and the occupancy rule; CAL2's "
            "census is a reported cross-check, and a difference is a STOP for the per-CTA readings"),
        "identification on DS's own pages (COMPUTED, if every unit runs)": D4.ds_identification(),
        "every unit nodrop": "a unit that does not run leaves readings NOT IDENTIFIED; DS is DESCRIPTIVE, so nothing blocks",
    }


def intent_block() -> dict:
    return {
        "label": "RECORDED INTENT, NOT CODE: nothing here runs, scores or gates; the next round builds and registers it",
        "sequence": ("after DS: read_ds.py's readings and CAL2's pages; then the next round chooses the decode model form, "
                     "freezes it and registers it with its gate before any page of Mixtral-8x22B, Qwen2-57B-A14B or "
                     "Granite-3.0-1B-A400M"),
        "future gate (owner, 2026-11-04)": (
            "the new blind runs ONLY on GO, and GO needs both: (1) CAL2 passes under the post-DS model (the addendum-3 "
            "stop rule on CAL2's kept cells); (2) the published decode blind pages re-score under the post-DS model at a "
            "pooled rms under 4.0%, labelled a WEAK SANITY FLOOR: those pages were seen during the diagnosis, so this "
            "clause is never a test and never a certification. Both published trees first reproduce their committed "
            "scores (reproduce_d4.py, built and kept here)"),
        "the new blind's bars (owner, 2026-11-04, D4-1)": "rms 3% per model, worst cell 8%, |mean| 1.5% (the CAL bars, "
                                                         "not tightened), to be registered with the model",
        "standing verdicts": "the decode blind FAIL (2026-11-04) and the decode CAL2 GO under addendum 3 stand",
    }


def reproduction_block() -> dict:
    return {
        "tool": f"scripts/scoring/decode/reproduce_d4.py {D4.CAL2_PUBLISHED}/results {D4.BLIND_PUBLISHED}/results",
        "rule": ("the CAL2 tree under v4d' (score_cal_d3's rules, frozen) reproduces the committed CAL2 score (the selected "
                 "rho form, T0_GR, s_naive, s_block and the in-sample E1 statistics), and the decode blind tree under the "
                 "committed finalisation (score_blind_d3.score, frozen) reproduces the committed blind score (E1-DEC pooled "
                 "and per model), each to 1e-9 relative. DESCRIPTIVE: it changes no verdict"),
    }


def seen_text() -> dict:
    t = D4.ds_touches()
    return {"what DS loads and times": t,
            "the new blind models (COMPUTED)": {m: D4.no_decode_page(m)["verdict"] for m in D4.NEW_BLIND},
            "what would make a new blind model SEEN": "any DS cell of its model, its weights, or a GEMM of its (H, I) "
                                                      "shape; DS has none (new_blind_models_loaded and "
                                                      "new_blind_shapes_measured are empty), and this addendum writes no "
                                                      "page of theirs",
            "SEEN already": ["the five earlier blind models' pages and the blind score (published 2026-11-04): SEEN, read "
                             "by both diagnoses; DS uses four of them (JetMoE, Phi, Granite-3B, Qwen1.5) and the two CAL "
                             "models", "the CAL2 session, its census and its score: SEEN"],
            "configs and generators read (COMPUTED, not pages)": D4.blind_reach(),
            "earlier non-decode pages of the new blind models (unused by addendum 4)": {
                "statement": ("'unseen' in this addendum means no decode page of the model exists. Earlier non-decode "
                              "GEMM pages of Qwen2-57B-A14B and Mixtral-8x22B are published and are not used by addendum "
                              "4: Qwen2-57B-A14B's settleS1 r3-counters kr and ar (2026-10-21) and cal3 stq "
                              "instrumented pages (2026-10-27), both models' 2026-09-28 GH200 sessions, and every other "
                              "session listed here (COMPUTED at b2fb3402 and frozen: a later publish changes nothing "
                              "here); Granite-3.0-1B-A400M's are listed the same way. No DS reading, plan or check reads "
                              "any of them"),
                "sessions (COMPUTED at b2fb3402, FROZEN)": D4.EARLIER_PAGES_B2FB3402}}


def build() -> dict:
    r = D4.rentals()
    return {
        "registered": f"{D4.DATE}, before any DS page",
        "name": D4.NAME,
        "status": "REGISTERED BEFORE ANY DS PAGE: the decode stage's fourth addendum, a MEASUREMENT registration only (no "
                  "model form, no constant, no gate); every earlier decode registration stays byte-identical (pinned "
                  "here); every registered verdict stands (the decode blind FAIL of 2026-11-04 included)",
        "design": DESIGN,
        "tool": f"after DS: scripts/scoring/decode/read_ds.py <DS tree> <out> (DESCRIPTIVE readings); "
                f"scripts/scoring/decode/reproduce_d4.py {D4.CAL2_PUBLISHED}/results {D4.BLIND_PUBLISHED}/results",
        "measurement registration (owner, 2026-11-04, D4-7)": measurement_block(),
        "settling rental DS (owner, 2026-11-04, D4-2a)": {
            "label": "DESCRIPTIVE: a rental of counter pages on SEEN models; every reading describes its pages",
            "census unit (owner, 2026-11-04, D4-8)": D4.census_line(),
            "units": D4.ds_units(),
            "plan": D4.DS_PLAN, "plan_sha256": RGD.sha256(REPO / D4.DS_PLAN),
            "driver": f"scripts/{D4.DS_DRIVER} (addendum 3's D3 copy, unchanged: its kernels=wide key runs "
                      "dram_counter_route_decode3.py --kernels wide; DECL arms run through the pinned counter child's "
                      "declared-copies path); no D4 driver copy, so vm_run.sh and docs/LAMBDA.md are unchanged",
            "pages": {f"{m}-{k}.json": D4.reg_file4(m, k) | {"what": D4.DS_WHAT} for m, k in D4.PAGE_KEYS},
            "estimates": "the wide route's rule (d3common): 0.6 s per profiled launch, 3 calls and 2 warmups profiled, 6 "
                         "launches per call, per (cell, arm), plus 2 min of setup per unit",
        },
        "stop rules (registered before any DS page)": stop_rules(),
        "readings (registered before any DS page)": readings_text(),
        "published-tree reproduction": reproduction_block(),
        "next round (RECORDED INTENT, not code; owner, 2026-11-04, D4-7)": intent_block(),
        "rentals": {
            "ds": r["ds"] | {"composition": f"prelude 3 + calibrate 2 + DS census {D4.CENSUS_EST} + six wide counter "
                                            "units " + " + ".join(str(u["est"]) for u in D4.ds_units())},
            "operator power step": "as addendum 2 (d2common.py --power on the prelude's power-set.txt; terminate unless READ)",
            "new blind": "not planned here: its plan, minutes and cost are registered with the model in the next round",
        },
        "seen": seen_text(),
        "unchanged": "the lock 1710 and G4 1705, the instruments, every registered page, the census rule, the tile source, "
                     "the card constants, v4d and v4d' (byte-identical), every pin of the five earlier decode "
                     "registrations, the drivers, vm_run.sh, docs/LAMBDA.md, and every registered verdict",
    }


def text(d: dict) -> list[str]:
    x = d["rentals"]["ds"]
    L = [f"REGISTERED {d['registered']}: the decode stage (a fourth addendum), {d['name']}", f"status: {d['status']}",
         f"design: {d['design']}", f"tool: {d['tool']}", ""]
    L.append(f"DS rental (settling, DESCRIPTIVE): {x['units_min']} min of units, about {x['wall_min']} min wall with "
             f"{x['boot_setup_min']} min of boot and setup: ${x['cost_usd']:.2f} at ${D2.RATE}/h; deadline launch + "
             f"{x['deadline_after_launch_min']} min; guardian hard cap launch + {x['hard_cap_after_launch_min']} min "
             f"(${x['cost_at_hard_cap_usd']:.2f} at most)")
    m = d["measurement registration (owner, 2026-11-04, D4-7)"]
    L.append(f"the model form: {m['the model form']}")
    L.append(f"disclosure: {m['disclosure (plain words)']}")
    L.append("DS units:")
    L.append("  " + d["settling rental DS (owner, 2026-11-04, D4-2a)"]["census unit (owner, 2026-11-04, D4-8)"])
    L += ["  " + D4.ds_line(u) for u in d["settling rental DS (owner, 2026-11-04, D4-2a)"]["units"]]
    L.append("")
    for k, v in d.items():
        if k in SKIP_TXT or k in ("registered", "name", "status", "design", "tool"):
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
    prev = json.loads((REG / f"{D3.NAME}.json").read_text())["code_pins"]
    paths = tuple(dict.fromkeys(tuple(prev) + FROZEN + RG3.D3_CODE + ("scripts/r3_timing_model_v4dp.py",
                                                                      "scripts/r3_timing_model_v4d.py") + SEEN_INPUTS
                                + D4_CODE + tuple(D4.page4_path(m, k) for m, k in D4.PAGE_KEYS)))
    return {p: RGD.sha256(REPO / p) for p in paths}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=REG)
    a = ap.parse_args(argv)
    files = D4.page4_files()
    pdir = REPO / D4.PAGES4
    plan = D4.ds_plan()
    if a.write:
        pdir.mkdir(parents=True, exist_ok=True)
        for n, t in files.items():
            (pdir / n).write_text(t)
        (REPO / D4.DS_PLAN).write_text(plan)
    doc = build()
    doc["code_pins"] = pins()
    jpath, tpath = a.out_dir / f"{D4.NAME}.json", a.out_dir / f"{D4.NAME}.txt"
    jtext = json.dumps(doc, indent=1, default=str) + "\n"
    ttext = "\n".join(text(doc)) + "\n"
    if a.check:
        strip = lambda x: {k: v for k, v in x.items() if k != "code_pins"}  # noqa: E731
        same_j = jpath.exists() and strip(json.loads(jpath.read_text())) == strip(json.loads(jtext))
        same_t = tpath.exists() and tpath.read_text() == ttext
        same_p = all((pdir / n).exists() and (pdir / n).read_text() == t for n, t in files.items())
        extra = sorted(p.name for p in pdir.glob("*.json") if p.name not in files) if pdir.is_dir() else []
        same_plan = (REPO / D4.DS_PLAN).exists() and (REPO / D4.DS_PLAN).read_text() == plan
        for name, same in ((jpath.name, same_j), (tpath.name, same_t), ("pages", same_p and not extra),
                           ("DS plan", same_plan)):
            print(f"{name}: {'same' if same else 'DIFFERS'}")
        if extra:
            print(f"pages not registered here: {', '.join(extra)}")
        return 0 if same_j and same_t and same_p and not extra and same_plan else 1
    jpath.write_text(jtext)
    tpath.write_text(ttext)
    print(f"wrote {jpath}\nwrote {tpath}\nwrote {len(files)} pages under {D4.PAGES4}\nwrote {D4.DS_PLAN}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
